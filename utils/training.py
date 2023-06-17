from __future__ import annotations

import pathlib
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from locale import D_FMT
from typing import TYPE_CHECKING, Any, Generator, Iterable, List, Tuple, Union
from unittest.mock import Base

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
class TrainingResults:
    """Stores the results of a training run."""

    rv: str = ""
    run_id: str = ""
    pred_ds: list = field(default_factory=list)
    res: float = 0.5
    params: dict = field(default_factory=dict)
    cv_nrmse: float = 0.0
    cv_nrmse_std: float = 0.0
    cv_r2: float = 0.0
    cv_r2_std: float = 0.0
    test_r2: float = 0.0
    predictors: list = field(default_factory=list)
    model_fn: pathlib.Path = field(default_factory=pathlib.Path)
    search_n_trials: int = 0
    cv_n_groups: int = 0
    cv_block_buffer: float = 0.0
    grid_size: float = 0.0
    random_state: int = 0

    def to_df(self) -> pd.DataFrame:
        """Converts the results to a pandas DataFrame."""
        df = pd.DataFrame(
            {
                "Run ID": self.run_id,
                "Response variable": self.rv,
                "Predictor datasets": [self.pred_ds],
                "Resolution": self.res,
                "Best parameters": [self.params],
                "N tuning iters": self.search_n_trials,
                "CV nRMSE": self.cv_nrmse,
                "CV nRMSE STD": self.cv_nrmse_std,
                "CV r-squared": self.cv_r2,
                "CV r-squared STD": self.cv_r2_std,
                "Test r-squared": self.test_r2,
                "Predictors": [self.predictors],
                "Model file": self.model_fn,
                "N CV groups": self.cv_n_groups,
                "CV grid size [m]": self.grid_size,
                "CV block buffer": self.cv_block_buffer,
                "Random seed": self.random_state,
            }
        )
        return df


@dataclass
class TrainingConfig:
    """
    Configuration options for a training run.

    Attributes:
        train_test_split (float, optional): Ratio of test set size to the
            whole dataset size. Defaults to 0.2.
        cv_grid_size (Union[int, float], optional): Size of the grid for
            spatial cross-validation. Defaults to 2330633.
        cv_n_groups (int, optional): Number of cross-validation groups.
            Defaults to 10.
        cv_block_buffer (float, optional): Buffer used in spatial blocking.
        search_n_trials (int, optional): Number of hyperparameter search
            trials. Defaults to 100.
        n_jobs (int, optional): Number of parallel jobs. Defaults to -1.
        results_dir (pathlib.Path, optional): Path to the directory to save
            training results. Defaults to "./results".
        results_csv (pathlib.Path, optional): Path to the CSV file where training
            results are saved. Defaults to results_dir / "training_results.csv".
        random_state (int, optional): Random seed. Defaults to 42.
    """

    train_test_split: float = 0.2
    cv_grid_size: Union[int, float] = 2330633
    cv_n_groups: int = 10
    cv_block_buffer: float = 0.0
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
        resume (bool, optional): Whether to resume the previous training session

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
        self.results.rv = self.y_col
        self.results.pred_ds = [dataset.id for dataset in XY.X.datasets]
        self.results.search_n_trials = self.training_config.search_n_trials
        self.results.cv_n_groups = self.training_config.cv_n_groups
        self.results.cv_block_buffer = self.training_config.cv_block_buffer
        self.results.grid_size = self.training_config.cv_grid_size
        self.random_state = self.training_config.random_state

        # Init a X an y dataframe with geometry, predictors, and target variable,
        # and drop rows and columns that contain only NA values
        # We need to combine the X and y dataframes to make sure that all y rows
        # that contain NA values are dropped in both the X and y dataframes
        Xy = self.XY.df[["geometry", *self.XY.X.cols, self.y_col]]
        Xy, X_cols, y_cols = drop_XY_NAs(Xy, XY.X.cols, y_col, True)

        # Split the data into X, y, and coordinates
        X = Xy[X_cols].to_numpy()
        y = Xy[y_col].to_numpy()
        coords = Xy["geometry"]

        # Get the 5th and 95th percentiles of the target variable to avoid outliers when
        # calculating the range (used for calculating the normalized RMSE)
        y_95 = np.percentile(y, 95)
        y_05 = np.percentile(y, 5)
        self.y_range = float(y_95 - y_05)

        # Split the data into training and testing sets
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
        if self.resume:
            # get last run ID from results CSV
            ids = pd.read_csv(self.training_config.results_csv)["run id"]
            id = ids.last_valid_index()  # ignores empty rows/NaNs
            if not id:
                warnings.warn("No previous training run found. Starting new run.")
            else:
                return ids.iloc[id]
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
        d = self.training_config.results_dir / "training" / self.id
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def spatial_cv(self) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
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

        self.results.params = params
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
            X_train=self.X_train,
            y_train=self.y_train,
            X_test=self.X_test,
            y_test=self.y_test,
            n_jobs=self.training_config.n_jobs,
        )

        model.save_model(self.results.model_fn)
        self.results.test_r2 = float(r2)

    def save_results(self) -> None:
        """Save results to csv and store dataframe on TrainingRun instance"""
        save_training_run_results(
            fn=self.training_config.results_csv,
            df=self.results.to_df(),
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
        grid_size (float): Size of grid in degrees
        n_groups (int, optional): Number of groups to split into. Defaults to 10.
        random_state (int, optional): Random state. Defaults to 42.
        verbose (int, optional): Verbosity. Defaults to 0.

    Returns:
        Iterable[Tuple[np.ndarray, np.ndarray]]: Spatial folds
    """
    if verbose == 1:
        print("Defining spatial folds...")
    tiles_x = int(np.round(360 / grid_size))
    tiles_y = int(np.round(180 / grid_size))

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
    X: np.ndarray,
    y: np.ndarray,
    col_name: str,
    cv: Union[BaseCrossValidator, Iterable[Tuple[np.ndarray, np.ndarray]]],
    save_dir: pathlib.Path,
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
