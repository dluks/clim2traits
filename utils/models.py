import ast
from dataclasses import dataclass

import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils.spatial_stats import aoa, block_cv_splits


@dataclass
class Stats:
    cv_nrmse: float
    cv_nrmse_std: float
    cv_r2: float
    cv_r2_std: float
    test_r2: float
    predictor_importances: list


@dataclass
class Model:
    id: int
    run_id: str
    y_name: str
    Xy: MLCollection
    resolution: float
    params: dict
    n_tuning_iters: int
    stats: Stats
    model_fpath: str
    random_state: int
    train_test_split: float
    n_cv_groups: int
    cv_grid_size: float
    cv_block_buffer: float
    # fold_indices: list
    n_observations: int
    optimizer: str
    max_iter: int
    filtered_y_outliers: list

    def __post_init__(self):
        tt_splits = train_test_split(
            self.Xy.df[self.Xy.X.cols].to_numpy(),
            self.Xy.df[self.Xy.Y.cols].to_numpy(),
            self.Xy.coords,
            test_size=self.train_test_split,
            random_state=self.random_state,
        )
        self.X_train = tt_splits[0]
        self.X_test = tt_splits[1]
        self.y_train = tt_splits[2]
        self.y_test = tt_splits[3]
        self.coords_train = tt_splits[4]
        self.coords_test = tt_splits[5]

    @property
    def model(self):
        model = xgb.XGBRegressor(**self.params)
        model.load_model(self.model_fpath)
        return model

    @property
    def cv(self):
        return block_cv_splits(
            X=self.X_train.to_numpy(),
            coords=self.coords_train,
            grid_size=self.cv_grid_size,
            buffer_radius=0,
            n_groups=self.n_cv_groups,
            random_state=self.random_state,
        )

    def plot_observed_vs_predicted(self):
        """Plot observed vs. predicted values."""
        pred = self.model.predict(self.X_test)
        obs = self.y_test

        # plot the observed vs. predicted values using seaborn
        sns.set_theme()
        sns.set_style("whitegrid")
        sns.set_context("paper")

        _, ax = plt.subplots()
        p1 = min(min(pred), min(obs))
        p2 = max(max(pred), max(obs))
        ax.plot([p1, p2], [p1, p2], color="black", lw=0.5)
        ax.scatter(pred, obs)

        # set informative axes and title
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Observed Values")
        ax.set_title(f"Observed vs. Predicted Values for\n{self.y_name}")
        plt.show()

    @classmethod
    def from_results_row(cls, row: pd.Series):
        predictor_ids = ast.literal_eval(row["Predictor datasets"])
        rv_ids = ast.literal_eval(row["RV datasets"])
        gbif_bands = [b.readable for b in GBIFBand]
        y_name = row["Response variable"]
        y_band = [b for b in gbif_bands if b in y_name][0]

        X = DataCollection([Dataset.from_id(id) for id in predictor_ids])
        Y = DataCollection([Dataset.from_id(id, y_band) for id in rv_ids])
        Y.df = Y.df[["geometry", y_name]]
        Xy = MLCollection(X, Y)
        Xy.drop_NAs(verbose=1)

        stats = Stats(
            cv_nrmse=row["CV nRMSE"],
            cv_nrmse_std=row["CV nRMSE STD"],
            cv_r2=row["CV r-squared"],
            cv_r2_std=row["CV r-squared STD"],
            test_r2=row["Test r-squared"],
            predictor_importances=ast.literal_eval(row["Predictor importance"]),
        )

        if isinstance(row["Filtered RV outliers"], str):
            filtered_y_outliers = ast.literal_eval(row["Filtered RV outliers"])
        else:
            filtered_y_outliers = []

        return cls(
            id=int(row.name.__str__()),
            run_id=row["Run ID"],
            y_name=y_name,
            Xy=Xy,
            resolution=row["Resolution"],
            params=ast.literal_eval(row["Best parameters"]),
            n_tuning_iters=int(row["N tuning iters"]),
            stats=stats,
            model_fpath=row["Model file"],
            train_test_split=0.2,  # TODO: add to results
            random_state=int(row["Random seed"]),
            n_cv_groups=int(row["N CV groups"]),
            cv_grid_size=row["CV grid size [m]"],
            cv_block_buffer=row["CV block buffer"],
            # fold_indices=ast.literal_eval(row["CV fold indices"]),
            n_observations=int(row["N observations"]),
            optimizer=row["Optimizer"],
            max_iter=int(row["max_iter"]),
            filtered_y_outliers=filtered_y_outliers,
        )


@dataclass
class Trait:
    name: str
    models: list[Model]
