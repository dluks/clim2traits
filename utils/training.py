import os
from datetime import datetime

import numpy as np
import spacv
from ray import tune
from ray.tune.sklearn import TuneSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split
from xgboost import XGBRegressor


def block_cv_splits(X, coords, autocorr_range, n_groups=10, random_state=0, verbose=0):
    if verbose == 1:
        print("Defining spatial folds...")
    DEGREE = 111325  # Standard value for 1 degree in meters at the equator
    tile = autocorr_range / DEGREE
    tiles_x = int(np.round(360 / tile))
    tiles_y = int(np.round(180 / tile))

    hblock = spacv.HBLOCK(
        tiles_x,
        tiles_y,
        shape="hex",
        method="optimized_random",
        buffer_radius=0.01,
        n_groups=n_groups,
        data=X,
        n_sims=50,
        distance_metric="haversine",
        random_state=random_state,
    )

    splits = hblock.split(coords)

    return splits


def optimize_params(
    X, y, col_name, cv, save_dir, n_trials=10, random_state=0, n_jobs=-1, verbose=0
):
    if verbose == 1:
        print("Optimizing parameters...")
    param_space = {
        "n_estimators": tune.lograndint(100, 2000),
        "max_depth": tune.randint(2, 6),
        "subsample": tune.quniform(0.25, 0.75, 0.01),
        "colsample_bytree": tune.quniform(0.05, 0.5, 0.01),
        "colsample_bylevel": tune.quniform(0.05, 0.5, 0.01),
        "learning_rate": tune.quniform(1e-3, 1e-1, 5e-4),
    }

    xgb_model = XGBRegressor(n_jobs=1, booster="gbtree")
    start_time = datetime.now()
    start = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{start}_{col_name}"

    reg = TuneSearchCV(
        xgb_model,
        param_space,
        n_trials=n_trials,
        scoring="neg_root_mean_squared_error",
        n_jobs=n_jobs,
        cv=cv,
        verbose=1,
        random_state=random_state,
        name=run_name,
        search_optimization="hyperopt",
        local_dir=save_dir,
    )
    reg.fit(X, y)
    params = reg.best_params_
    rmse = -reg.best_score_
    std = reg.cv_results_["std_test_score"][reg.best_index_]
    return params, rmse, std


def train_model_cv(model_params, X, y, cv, n_jobs=-1, verbose=0):
    if verbose == 1:
        print("Assessing model performance with spatial CV...")
    print("Params:", model_params)
    print("Params type:", type(model_params))
    model = XGBRegressor(**model_params, booster="gbtree")
    scores = -cross_val_score(
        model,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        scoring="neg_root_mean_squared_error",
    )
    mean_rmse, std = np.sqrt(scores).mean(), np.sqrt(scores).std()
    return mean_rmse, std, scores


def train_model_full(
    model_params, X_train, y_train, X_test, y_test, verbose=0, n_jobs=-1
):
    if verbose == 1:
        print("Training full model and saving...")

    model = XGBRegressor(
        **model_params,
        booster="gbtree",
        early_stopping_rounds=100,
        verbose=0,
        eval_metric=mean_squared_error,
        n_jobs=n_jobs,
    )
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Evaluate prediction on test data
    r2 = model.score(X_test, y_test)

    return model, r2


def run_training(
    Xy,
    X_cols,
    y_col,
    autocorr_range,
    search_n_trials,
    n_jobs,
    random_state,
    param_opt_save_dir,
    final_save_dir,
    run_name,
    steps=[1, 2, 3],
):
    # Define results for optional returning
    rmse = None
    std = None
    r2 = None
    model_fn = None
    params = None

    # 0. Separate X, y, and coords and split into train/test groups
    print("\nSplitting into train/test...")
    X = Xy[X_cols].to_numpy()
    y = Xy[y_col].to_numpy()
    coords = Xy["geometry"]

    X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
        X, y, coords, test_size=0.2
    )
    if 1 in steps:
        # 1. Spatially block data into cells according to the autocorrelation range and
        #    separate them into K folds
        spatial_cv = block_cv_splits(
            X=X_train,
            coords=coords_train,
            autocorr_range=autocorr_range,
            n_groups=10,
            random_state=random_state,
        )

    if 2 in steps:
        # 2. Identify optimal hyperparameters and get best performance
        params, rmse, std = optimize_params(
            X=X_train,
            y=y_train,
            col_name=y_col,
            cv=spatial_cv,
            n_trials=search_n_trials,
            save_dir=param_opt_save_dir,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    # if 3 in steps:
    #     # 3. Get model performance with spatial CV
    #     # Get CV iterator again
    #     spatial_cv = block_cv_splits(
    #         X_train, coords_train, autocorr_range, random_state
    #     )

    #     mean_rmse, std, scores = train_model_cv(
    #         params, X_train, y_train, spatial_cv, n_jobs=1
    #     )

    if 3 in steps:
        # 4. Train full model and save model and R^2
        model_fn = os.path.join(final_save_dir, run_name)

        model, r2 = train_model_full(
            model_params=params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            n_jobs=n_jobs,
        )

        model.save_model(model_fn)

    return model_fn, params, rmse, std, r2
