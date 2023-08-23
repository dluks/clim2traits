from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from typing import TYPE_CHECKING, Iterable, Optional
from typing import SupportsFloat as Numeric
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from ray import tune
from ray.tune.sklearn import TuneSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import BaseCrossValidator, cross_validate, train_test_split
from xgboost import XGBRegressor

from utils.spatial_stats import block_cv_splits

if TYPE_CHECKING:
    from utils.datasets import MLCollection

from utils.geodata import drop_XY_NAs


@dataclass
class TrainingResults:
    """Stores the results of a training run."""

    rv: str = ""
    n_obs: int = 0
    run_id: str = ""
    rv_ds: list = field(default_factory=list)
    pred_ds: list = field(default_factory=list)
    collection_fn: Optional[str] = None
    res: str = "0.5_deg"
    params: dict = field(default_factory=dict)
    cv_nrmse: Numeric = 0.0
    cv_nrmse_std: Numeric = 0.0
    cv_r2: Numeric = 0.0
    cv_r2_std: Numeric = 0.0
    test_r2: Numeric = 0.0
    predictor_importance: list = field(default_factory=list)
    model_fn: pathlib.Path = field(default_factory=pathlib.Path)
    search_n_trials: int = 0
    optimizer: str = ""
    max_iters: int = 0
    cv_n_groups: int = 0
    cv_block_buffer: Numeric = 0.0
    grid_size: Numeric = 0.0
    fold_indices: list = field(default_factory=list)
    random_state: int = 0
    filtered_outliers: Optional[list] = None
    nan_strategy: str = "all"

    def to_df(self) -> pd.DataFrame:
        """Converts the results to a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "Run ID": self.run_id,
                "Response variable": self.rv,
                "N observations": self.n_obs,
                "RV datasets": [self.rv_ds],
                "Predictor datasets": [self.pred_ds],
                "collection": self.collection_fn,
                "Resolution": self.res,
                "Best parameters": [self.params],
                "Optimizer": self.optimizer,
                "max_iter": self.max_iters,
                "N tuning iters": self.search_n_trials,
                "CV nRMSE": self.cv_nrmse,
                "CV nRMSE STD": self.cv_nrmse_std,
                "CV r-squared": self.cv_r2,
                "CV r-squared STD": self.cv_r2_std,
                "Test r-squared": self.test_r2,
                "Predictor importance": [self.predictor_importance],
                "Model file": self.model_fn,
                "N CV groups": self.cv_n_groups,
                "CV grid size [m]": self.grid_size,
                "CV block buffer": self.cv_block_buffer,
                "CV fold indices": [[*self.fold_indices]],
                "Random seed": self.random_state,
                "Filtered RV outliers": [self.filtered_outliers],
                "NaN strategy": self.nan_strategy,
            }
        )
        return df


@dataclass
class TrainingConfig:
    """
    Configuration options for a training run.

    Attributes:
        nan_strategy (str, optional): Strategy for handling NaNs in the data.
        train_test_split (float, optional): Ratio of test set size to the
            whole dataset size. Defaults to 0.2.
        cv_grid_size (Union[int, float], optional): Size of the grid for
            spatial cross-validation. Defaults to 2330633.
        cv_n_groups (int, optional): Number of cross-validation groups.
            Defaults to 10.
        cv_block_buffer (float, optional): Buffer used in spatial blocking.
        search_n_trials (int, optional): Number of hyperparameter search
            trials. Defaults to 100.
        optimizer (str, optional): Hyperparameter optimization algorithm.
            Defaults to "hyperopt".
        params (dict, optional): Hyperparameter search space. Defaults to {}.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        results_dir (pathlib.Path, optional): Path to the directory to save
            training results. Defaults to "./results".
        results_csv (pathlib.Path, optional): Path to the CSV file where training
            results are saved. Defaults to results_dir / "training_results.csv".
        random_state (int, optional): Random seed. Defaults to 42.
        filter_y_outliers (Optional[list], optional): Quantile range with which to
            filter outliers in the response variable, e.g. [0.5, 0.95] Defaults to None.
    """

    nan_strategy: str = "all"
    train_test_split: float = 0.2
    cv_grid_size: Union[int, float] = 2330633
    cv_n_groups: int = 10
    cv_block_buffer: float = 0.0
    search_n_trials: int = 100
    optimizer: str = "hyperopt"
    params: dict = field(default_factory=dict)
    max_iters: int = 1
    n_jobs: int = -1
    results_dir: pathlib.Path = pathlib.Path("./results")
    results_csv: pathlib.Path = pathlib.Path(results_dir) / "training_results.csv"
    random_state: int = 42
    filter_y_outliers: Optional[list] = None


class TrainingRun:
    """
    Represents a training run for an XGBoost regression model. It incorporates
    hyperparameter optimization with spatial cross-validation, and training of the model
    on the full training dataset with final predictions on the test set.

    Args:
        XY (MLCollection): Collection of all X and Y data.
        y_col (str): Name of target response variable.
        training_config (TrainingConfig): Training configuration.
        resume (bool, optional): Whether to resume the previous training session

    Attributes:
        XY (MLCollection): Collection of all X and Y data.
        y_col (str): Name of target response variable.
        training_config (TrainingConfig): Training configuration.
        results (dict): Dictionary to store the results of the training run.
        X (ndarray): Input data.
        y (ndarray): Target data.
        coords_train (ndarray): Coordinates for the training data.
        coords_test (ndarray): Coordinates for the testing data.

    Properties:
        id (str): Unique identifier for this training run.
        hyperopt_dir (str): Directory to save hyperopt results.
        results_dir (str): Directory to save training results.
        spatial_cv (Generator[Tuple[NDArray, NDArray]]): Spatial cross-validation
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
        resume: bool = False,
    ) -> None:
        """Initialize TrainingRun with a MLCollection, target response variable, and
        training configuration.

        Args:
            XY (MLCollection): Collection of all X and Y data
            y_col (str): Name of target response variable
            training_config (TrainingConfig): Training configuration
            resume (bool, optional): Whether to resume a previous training session
                (i.e. use the last training run ID).
        """
        self.XY = XY
        self.y_col = y_col
        self.training_config = training_config
        self.resume = resume
        self.results = TrainingResults()  # Initialize TrainingResults object

        # Store the IDs of the datasets used as predictors
        self.results.rv_ds = [dataset.id for dataset in XY.Y.datasets]
        self.results.pred_ds = [dataset.id for dataset in XY.X.datasets]
        if XY.X.collection_file is not None:
            self.results.collection_fn = str(XY.X.collection_file)
        self.results.rv = self.y_col
        self.results.run_id = self.id
        self.results.res = self.XY.X.datasets[0].res_str
        self.results.params = self.training_config.params
        self.results.search_n_trials = self.training_config.search_n_trials
        self.results.optimizer = self.training_config.optimizer
        self.results.max_iters = self.training_config.max_iters
        self.results.cv_n_groups = self.training_config.cv_n_groups
        self.results.cv_block_buffer = self.training_config.cv_block_buffer
        self.results.grid_size = self.training_config.cv_grid_size
        self.results.random_state = self.training_config.random_state
        self.results.filtered_outliers = self.training_config.filter_y_outliers
        self.results.nan_strategy = self.training_config.nan_strategy

        # Init a X an y dataframe with geometry, predictors, and target variable,
        # and drop rows and columns that contain only NA values
        # We need to combine the X and y dataframes to make sure that all y rows
        # that contain NA values are dropped in both the X and y dataframes
        Xy = self.XY.df[["geometry", *self.XY.X.cols, self.y_col]]
        Xy, X_cols, y_cols = drop_XY_NAs(Xy, XY.X.cols, y_col, True)
        self.predictor_names = X_cols.values
        self.results.n_obs = len(Xy)

        # Split the data into X, y, and coordinates
        self.X = Xy[X_cols].to_numpy()
        self.y = Xy[y_col].to_numpy()
        self.coords = Xy["geometry"]

        # Get the 5th and 95th percentiles of the target variable to avoid outliers when
        # calculating the range (used for calculating the normalized RMSE)
        y_95 = np.percentile(self.y, 95)
        y_05 = np.percentile(self.y, 5)
        self.y_range = float(y_95 - y_05)

        # Split the data into training and testing sets (not really sure why I did this
        # anymore)
        tt_splits = train_test_split(
            self.X,
            self.y,
            self.coords,
            test_size=self.training_config.train_test_split,
            random_state=self.training_config.random_state,
        )
        self.X_train = tt_splits[0]
        self.X_test = tt_splits[1]
        self.y_train = tt_splits[2]
        self.y_test = tt_splits[3]
        self.coords_train = tt_splits[4]
        self.coords_test = tt_splits[5]

    @cached_property
    def id(self) -> str:
        """Unique identifier for this training run"""
        if self.resume:
            # get last run ID from results CSV
            ids = pd.read_csv(self.training_config.results_csv)["Run ID"].sort_values()
            id = ids.last_valid_index()  # ignores empty rows/NaNs
            if id is None:
                warnings.warn("No previous training run found. Starting new run.")
            else:
                return ids.iloc[id]  # type: ignore
        return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @property
    def hyperopt_dir(self) -> pathlib.Path:
        """Directory to save hyperopt results"""
        d = self.training_config.results_dir / "ray-results" / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def results_dir(self) -> pathlib.Path:
        """Directory to save training results"""
        d = self.training_config.results_dir / "training" / self.id / self.y_col
        d.mkdir(parents=True, exist_ok=True)
        return d

    @cached_property
    def spatial_cv(self) -> Iterable[Tuple[npt.NDArray, npt.NDArray]]:
        """Spatial CV iterator"""
        cv = block_cv_splits(
            X=self.X,
            coords=self.coords,
            grid_size=self.training_config.cv_grid_size,
            buffer_radius=self.training_config.cv_block_buffer,
            n_groups=self.training_config.cv_n_groups,
            random_state=self.training_config.random_state,
        )
        cv = list(cv)
        folds = [fold[1] for fold in cv]
        # Save fold indices to results object (need to convert nested numpy arrays to
        # lists so that they can be evaluated with ast later)
        self.results.fold_indices = [fold.tolist() for fold in folds]
        return cv

    def tune_params_cv(self):
        """Tune model hyperparameters with spatial CV and save results"""
        reg = optimize_params(
            X=self.X,
            y=self.y,
            col_name=self.y_col,
            cv=self.spatial_cv,
            n_trials=self.training_config.search_n_trials,
            optimizer=self.training_config.optimizer,
            max_iters=self.training_config.max_iters,
            save_dir=self.hyperopt_dir,
            random_state=self.training_config.random_state,
            n_jobs=self.training_config.n_jobs,
        )
        params = reg.best_params
        rmses, r2s = get_cv_results(reg, self.training_config.cv_n_groups)
        cv_nrmse, cv_nrmse_std = normalize_to_range(rmses, self.y_range)
        cv_r2, cv_r2_std = r2s.mean(), r2s.std()

        self.results.params = params
        self.results.cv_nrmse = cv_nrmse
        self.results.cv_nrmse_std = cv_nrmse_std
        self.results.cv_r2 = cv_r2
        self.results.cv_r2_std = cv_r2_std

    def train_cv(self) -> None:
        """Train model with spatial CV and save results"""
        cv_results = train_model_cv(
            model_params=self.results.params,
            X=self.X,
            y=self.y,
            cv=self.spatial_cv,
            n_jobs=self.training_config.n_jobs,
            verbose=1,
        )

        # Save each CV estimator to disk
        estimators = cv_results["estimator"]
        cv_est_out_dir = self.results_dir / "cv-estimators"
        cv_est_out_dir.mkdir(parents=True, exist_ok=True)

        for i, est in enumerate(estimators):
            est_fn = cv_est_out_dir / f"{self.y_col}_{self.id}_fold-{i}"
            est.save_model(est_fn)

        rmses, r2s = cv_results["test_rmse"], cv_results["test_r2"]
        cv_nrmse, cv_nrmse_std = normalize_to_range(-rmses, self.y_range)
        cv_r2, cv_r2_std = r2s.mean(), r2s.std()

        self.results.cv_nrmse = cv_nrmse
        self.results.cv_nrmse_std = cv_nrmse_std
        self.results.cv_r2 = cv_r2
        self.results.cv_r2_std = cv_r2_std

    def train_model_on_all_data(self) -> None:
        """
        Train model on all data with optimal hyperparameters and save model and
        results
        """
        self.results.model_fn = self.results_dir / f"{self.y_col}_{self.id}"

        model, r2 = train_model_full(
            model_params=self.results.params,
            X=self.X,
            y=self.y,
            n_jobs=self.training_config.n_jobs,
            random_state=self.training_config.random_state,
        )

        model.save_model(self.results.model_fn)
        ft_imp = np.column_stack((self.predictor_names, model.feature_importances_))
        # reverse sort by importance (highest to lowest)
        self.results.predictor_importance = ft_imp[
            ft_imp[:, 1].argsort()[::-1]
        ].tolist()
        self.results.test_r2 = r2

    def save_results(self) -> None:
        """Save results to csv and store dataframe on TrainingRun instance"""
        save_training_run_results(
            fn=self.training_config.results_csv,
            df=self.results.to_df(),
        )


def normalize_to_range(x: npt.NDArray, range: float) -> Tuple[float, float]:
    """Normalize a dataset to a range

    Args:
        x (NDArray): Data
        range (float): Range to normalize with

    Returns:
        tuple: Mean normalized dataset and its standard deviation
    """
    return float(np.mean(x / range)), float(np.std(x / range))


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
    X: npt.NDArray,
    y: npt.NDArray,
    col_name: str,
    cv: Union[BaseCrossValidator, Iterable[Tuple[npt.NDArray, npt.NDArray]]],
    save_dir: pathlib.Path,
    n_trials: int = 10,
    optimizer: str = "hyperopt",
    max_iters: int = 1,
    random_state: int = 0,
    n_jobs: int = -1,
    verbose: int = 0,
) -> tuple:
    """Optimize XGBoost model hyperparameters using Ray Tune and Hyperopt.

    Args:
        X (NDArray): Input data.
        y (NDArray): Target data.
        col_name (str): Name of the target column.
        cv (BaseCrossValidator): Cross-validation object or iterator.
        save_dir (pathlib.Path): Directory to save optimization results.
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
        "n_estimators": tune.lograndint(255, 1273),
        "max_depth": tune.randint(5, 10),
        "subsample": tune.quniform(0.45, 0.7, 0.005),
        "colsample_bytree": tune.quniform(0.2, 0.65, 0.01),
        "colsample_bylevel": tune.quniform(0.2, 0.45, 0.005),
        "learning_rate": tune.quniform(0.01, 0.05, 0.0005),
    }

    xgb_model = XGBRegressor(n_jobs=1, booster="gbtree")

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
        name=col_name,
        search_optimization=optimizer,
        max_iters=max_iters,
        local_dir=str(save_dir.absolute()),
    )
    reg.fit(X, y)

    return reg


def train_model_cv(
    model_params: dict,
    X: npt.NDArray,
    y: npt.NDArray,
    cv: Union[BaseCrossValidator, Iterable[Tuple[npt.NDArray, npt.NDArray]]],
    n_jobs: int = -1,
    verbose: int = 0,
) -> dict:
    """Train the model with cross-validation and return performance metrics.

    Args:
        model_params (dict): Parameters for the XGBoost model.
        X (NDArray): Input data.
        y (NDArray): Target data.
        cv: Cross-validation object or iterator.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        tuple: A tuple containing the mean RMSE, standard deviation, and individual scores.
    """
    if verbose == 1:
        print("Assessing model performance with spatial CV...")

    print("Params:", model_params)

    model = XGBRegressor(**model_params, booster="gbtree", n_jobs=1)

    # scores = -cross_val_score(
    #     model,
    #     X,
    #     y,
    #     cv=cv,
    #     n_jobs=n_jobs,
    #     verbose=verbose,
    #     scoring="neg_root_mean_squared_error",
    # )

    cv_results = cross_validate(
        estimator=model,
        X=X,
        y=y,
        scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
        cv=cv,
        return_estimator=True,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    # mean_rmse, std = np.sqrt(scores).mean(), np.sqrt(scores).std()

    # return mean_rmse, std, scores
    return cv_results


def train_model_full(
    model_params: dict,
    X: npt.NDArray,
    y: npt.NDArray,
    verbose: int = 0,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[XGBRegressor, Numeric]:
    if verbose == 1:
        print("Training full model and saving...")

    model = XGBRegressor(
        **model_params,
        booster="gbtree",
        importance_type="gain",
        n_jobs=n_jobs,
        random_state=random_state,
    )
    model.fit(X, y, verbose=False)

    # Evaluate prediction on test data
    r2 = model.score(X, y)

    return model, r2


def save_training_run_results(
    fn: pathlib.Path,
    df: pd.DataFrame,
) -> None:
    """Save training results to csv and return dataframe

    Args:
        fn (pathlib.Path): Path to CSV file (will be created if it doesn't exist)
        df (pd.DataFrame): Dataframe containing training results
    """

    if fn.is_file():
        old_df = pd.read_csv(fn)
        old_df.dropna(how="all", inplace=True)  # clean up empty rows
        comb_df = pd.concat([old_df, df], ignore_index=True)
        comb_df.to_csv(fn, index=False)
    else:
        df.to_csv(fn, index=False)
