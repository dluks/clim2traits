from __future__ import annotations

import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Generator, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
import spacv
from ray import tune
from ray.tune.sklearn import TuneSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    BaseCrossValidator,
    cross_val_score,
    train_test_split,
)
from xgboost import XGBRegressor

if TYPE_CHECKING:
    from utils.datasets import MLCollection

from utils.geodata import drop_XY_NAs


@dataclass
class TrainingConfig:
    """
    Configuration options for a training run.

    Args:
        train_test_split (float, optional): Ratio of test set size to the
            whole dataset size. Defaults to 0.2.
        cv_grid_size (Union[int, float], optional): Size of the grid for
            spatial cross-validation. Defaults to 2330633.
        cv_n_groups (int, optional): Number of cross-validation groups.
            Defaults to 10.
        random_state (int, optional): Random seed. Defaults to 42.
        search_n_trials (int, optional): Number of hyperparameter search
            trials. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        results_dir (pathlib.Path, optional): Path to the directory to save
            training results. Defaults to "./results".

    Attributes:
        train_test_split (float): Ratio of test set size to the whole dataset
            size.
        cv_grid_size (Union[int, float]): Size of the grid for spatial
            cross-validation.
        cv_n_groups (int): Number of cross-validation groups.
        random_state (int): Random seed.
        search_n_trials (int): Number of hyperparameter search trials.
        n_jobs (int): Number of parallel jobs.
        results_dir (pathlib.Path): Path to the directory to save training
            results.
    """

    train_test_split: float = 0.2
    cv_grid_size: Union[int, float] = 2330633
    cv_n_groups: int = 10
    search_n_trials: int = 100
    n_jobs: int = -1
    results_dir: pathlib.Path = pathlib.Path("./results")
    results_csv: pathlib.Path = pathlib.Path(results_dir) / "training_results.csv"
    random_state: int = 42


class TrainingRun:
    """
    Represents a training run for an XGBoost regression model. It incorporates
    hyperparameter optimization with spatial cross-validation, and training of the model
    on the full training dataset with final predictions on the test set.

    Args:
        XY (MLCollection): Collection of all X and Y data.
        y_col (str): Name of target response variable.
        training_config (TrainingConfig): Training configuration.

    Attributes:
        XY (MLCollection): Collection of all X and Y data.
        y_col (str): Name of target response variable.
        training_config (TrainingConfig): Training configuration.
        results (dict): Dictionary to store the results of the training run.
        X_train (ndarray): Training input data.
        X_test (ndarray): Testing input data.
        y_train (ndarray): Training target data.
        y_test (ndarray): Testing target data.
        coords_train (ndarray): Coordinates for the training data.
        coords_test (ndarray): Coordinates for the testing data.

    Properties:
        id (str): Unique identifier for this training run.
        hyperopt_dir (str): Directory to save hyperopt results.
        results_dir (str): Directory to save training results.
        spatial_cv (Generator[Tuple[np.ndarray, np.ndarray]]): Spatial cross-validation
            iterator.

    Methods:
        tune_params_cv(): Tune model hyperparameters with spatial cross-validation.
        train_model_on_all_data(): Train model on all data with optimal hyperparameters.
        save_results(): Save results to CSV.
    """

    def __init__(
        self,
        XY: MLCollection,
        y_col: str,
        training_config: TrainingConfig,
    ) -> None:
        """Initialize TrainingRun with a MLCollection, target response variable, and
        training configuration.

        Args:
            XY (MLCollection): Collection of all X and Y data
            y_col (str): Name of target response variable
            training_config (TrainingConfig): Training configuration
        """
        self.XY = XY
        self.y_col = y_col
        self.training_config = training_config

        self.results = {}  # initialize results dict

        Xy = self.XY.df[["geometry", *self.XY.X.cols, self.y_col]]
        Xy, X_cols, y_cols = drop_XY_NAs(Xy, XY.X.cols, y_col, True)

        self.predictor_datasets = [dataset.id for dataset in XY.X.datasets]

        X = Xy[X_cols].to_numpy()
        y = Xy[y_col].to_numpy()
        coords = Xy["geometry"]

        y_95 = np.percentile(y, 95)
        y_05 = np.percentile(y, 5)

        self.y_range = y_95 - y_05

        tt_splits = train_test_split(
            X, y, coords, test_size=self.training_config.train_test_split
        )

        self.X_train = tt_splits[0]
        self.X_test = tt_splits[1]
        self.y_train = tt_splits[2]
        self.y_test = tt_splits[3]
        self.coords_train = tt_splits[4]
        self.coords_test = tt_splits[5]

    @property
    def id(self) -> str:
        """Unique identifier for this training run"""
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def hyperopt_dir(self) -> str:
        """Directory to save hyperopt results"""
        d = self.training_config.results_dir / "ray-results" / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_dir(self) -> str:
        """Directory to save training results"""
        d = self.training_config.results_dir / "training" / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def spatial_cv(self) -> Generator[Tuple[np.ndarray, np.ndarray]]:
        """Spatial CV iterator"""
        return block_cv_splits(
            X=self.X_train,
            coords=self.coords_train,
            grid_size=self.training_config.cv_grid_size,
            n_groups=self.training_config.cv_n_groups,
            random_state=self.training_config.random_state,
        )

    def tune_params_cv(self):
        """Tune model hyperparameters with spatial CV and save results"""
        reg = optimize_params(
            X=self.X_train,
            y=self.y_train,
            col_name=self.y_col,
            cv=self.spatial_cv,
            n_trials=self.training_config.search_n_trials,
            save_dir=self.hyperopt_dir,
            random_state=self.training_config.random_state,
            n_jobs=self.training_config.n_jobs,
        )
        params = reg.best_params

        rmses, r2s = get_cv_results(reg, self.training_config.cv_n_groups)

        cv_nrmse, cv_nrmse_std = normalize_to_range(rmses, self.y_range)

        cv_r2, cv_r2_std = r2s.mean(), r2s.std()

        self.results["params"] = params
        self.results["cv_nrmse"] = cv_nrmse
        self.results["cv_nrmse_std"] = cv_nrmse_std
        self.results["cv_r2"] = cv_r2
        self.results["cv_r2_std"] = cv_r2_std

    def train_model_on_all_data(self) -> None:
        """
        Train model on all data with optimal hyperparameters and save model and
        results
        """
        self.model_fn = self.results_dir / f"{self.y_col}_{self.id}"

        model, r2 = train_model_full(
            model_params=self.results["params"],
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            n_jobs=self.training_config.n_jobs,
        )

        model.save_model(self.model_fn)
        self.results["model_fn"] = self.model_fn
        self.results["r2"] = r2

    def save_results(self) -> None:
        """Save results to csv and store dataframe on TrainingRun instance"""
        self.results_df = save_training_run_results(
            save_dir=self.training_config.results_dir,
            run_id=self.id,
            response_variable=self.y_col,
            resolution=self.XY.Y.datasets[0].res_str,
            params=self.results["params"],
            cv_nrmse=self.results["cv_nrmse"],
            cv_nrmse_std=self.results["cv_nrmse_std"],
            cv_r2=self.results["cv_r2"],
            cv_r2_std=self.results["cv_r2_std"],
            test_r2=self.results["r2"],
            predictors=self.XY.X.cols.values,
            predictor_datasets=self.predictor_datasets,
            model_fn=pathlib.Path(self.results["model_fn"]),
        )


def block_cv_splits(
    X: np.ndarray,
    coords: pd.Series,
    grid_size: float,
    n_groups: int = 10,
    random_state: int = 42,
    verbose: int = 0,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """Define spatial folds for cross-validation

    Args:
        X (np.ndarray): X training data
        coords (pd.Series): Coordinates of training data
        grid_size (float): Size of grid in meters
        n_groups (int, optional): Number of groups to split into. Defaults to 10.
        random_state (int, optional): Random state. Defaults to 42.
        verbose (int, optional): Verbosity. Defaults to 0.

    Returns:
        Iterable[Tuple[np.ndarray, np.ndarray]]: Spatial folds
    """
    if verbose == 1:
        print("Defining spatial folds...")
    DEGREE = 111325  # Standard value for 1 degree in meters at the equator
    tile = grid_size / DEGREE
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


def normalize_to_range(x: np.ndarray, range: float) -> Tuple[float, float]:
    """Normalize a dataset to a range

    Args:
        x (np.ndarray): Data
        range (float): Range to normalize with

    Returns:
        tuple: Mean normalized dataset and its standard deviation
    """
    return np.mean(x / range), np.std(x / range)


def get_cv_results(reg: TuneSearchCV, nsplits: int):
    """Get cross-validation results from a TuneSearchCV object"""
    cv_results = reg.cv_results_
    rmses = np.empty(nsplits)
    r2s = np.empty(nsplits)

    for i in list(range(nsplits)):
        rmses[i] = -cv_results[f"split{i}_test_rmse"][reg.best_index_]
        r2s[i] = cv_results[f"split{i}_test_r2"][reg.best_index_]

    return rmses, r2s


def optimize_params(
    X: np.ndarray,
    y: np.ndarray,
    col_name: str,
    cv: BaseCrossValidator,
    save_dir: str,
    n_trials: int = 10,
    random_state: int = 0,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple:
    """Optimize XGBoost model hyperparameters using Ray Tune and Hyperopt.

    Args:
        X (np.ndarray): Input data.
        y (np.ndarray): Target data.
        col_name (str): Name of the target column.
        cv (BaseCrossValidator): Cross-validation object or iterator.
        save_dir (str): Directory to save optimization results.
        n_trials (int, optional): Number of trials. Defaults to 10.
        random_state (int, optional): Random seed. Defaults to 0.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        TuneSearchCV: Optimized model.
    """
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
    save_dir = pathlib.Path(save_dir, "ray-results")
    save_dir.mkdir(parents=True, exist_ok=True)

    reg = TuneSearchCV(
        xgb_model,
        param_space,
        n_trials=n_trials,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        refit="rmse",
        n_jobs=n_jobs,
        cv=cv,
        verbose=1,
        random_state=random_state,
        name=run_name,
        search_optimization="hyperopt",
        local_dir=str(save_dir.absolute()),
    )
    reg.fit(X, y)

    return reg


def train_model_cv(
    model_params: dict,
    X: np.ndarray,
    y: np.ndarray,
    cv,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple:
    """Train the model with cross-validation and return performance metrics.

    Args:
        model_params (dict): Parameters for the XGBoost model.
        X (np.ndarray): Input data.
        y (np.ndarray): Target data.
        cv: Cross-validation object or iterator.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        tuple: A tuple containing the mean RMSE, standard deviation, and individual scores.
    """
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


def save_training_run_results(
    save_dir: pathlib.Path,
    run_id: str,
    response_variable: str,
    resolution: str,
    params: dict,
    cv_nrmse: float,
    cv_nrmse_std: float,
    cv_r2: float,
    cv_r2_std: float,
    test_r2: float,
    predictors: List[str],
    predictor_datasets: List[str],
    model_fn: pathlib.Path,
) -> pd.DataFrame:
    """Save training results to csv and return dataframe

    Args:
        save_dir (pathlib.Path): Directory in which to save results
        run_id (str): Unique identifier for this training run
        response_variable (str): Response variable name
        resolution (str): Resolution of dataset
        params (dict): Best model hyperparameters
        cv_mean_rmse (float): Mean RMSE from spatial CV
        cv_std (float): Standard deviation of mean RMSE from spatial CV
        cv_mean_n_rmse (float): Mean normalized RMSE from spatial CV
        r2 (float): R-squared of model on test data
        predictors (List[str]): X variable names
        predictor_datasets (List[str]): X dataset names
        model_fn (pathlib.Path): Path to saved model

    Returns:
        pd.DataFrame:
    """
    results_df = pd.DataFrame(
        {
            "run id": run_id,
            "response variable": response_variable,
            "predictor datasets": [predictor_datasets],
            "resolution": resolution,
            "params": [params],
            "CV nRMSE": cv_nrmse,
            "CV nRMSE STD": cv_nrmse_std,
            "CV r-squared": cv_r2,
            "CV r-squared STD": cv_r2_std,
            "Test r-squared": test_r2,
            "predictors": [predictors],
            "model file": model_fn,
        }
    )

    csv = save_dir / "training_results.csv"

    if csv.is_file():
        old_df = pd.read_csv(csv)
        old_df = pd.concat([old_df, results_df], ignore_index=True)
        old_df.to_csv(csv, index=False)
    else:
        results_df.to_csv(csv, index=False)

    return results_df
