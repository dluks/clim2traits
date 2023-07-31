import ast
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

from utils.datasets import DataCollection, GBIFBand, MLCollection
from utils.spatial_stats import aoa, block_cv_splits


@dataclass
class Stats:
    cv_nrmse: float
    cv_nrmse_std: float
    cv_r2: float
    cv_r2_std: float
    test_r2: float
    predictor_importances: np.ndarray


@dataclass
class TrainedSet:
    id: int
    run_id: str
    y_name: str
    Xy: MLCollection
    Xy_imputed: Optional[MLCollection]
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
            self.Xy.df,
            test_size=self.train_test_split,
            random_state=self.random_state,
        )

        x_cols = self.Xy.X.cols
        y_cols = self.Xy.Y.cols

        self.X_train = tt_splits[0][x_cols]
        self.X_test = tt_splits[1][x_cols]
        self.y_train = tt_splits[0][y_cols]
        self.y_test = tt_splits[1][y_cols]
        self.coords_train = tt_splits[0]["geometry"]
        self.coords_test = tt_splits[1]["geometry"]

        if self.Xy_imputed is not None:
            tt_splits = train_test_split(
                self.Xy_imputed.df,
                test_size=self.train_test_split,
                random_state=self.random_state,
            )

            self.X_train_imputed = tt_splits[0][x_cols]
            self.X_test_imputed = tt_splits[1][x_cols]
            self.y_train_imputed = tt_splits[0][y_cols]
            self.y_test_imputed = tt_splits[1][y_cols]
            self.coords_train_imputed = tt_splits[0]["geometry"]
            self.coords_test_imputed = tt_splits[1]["geometry"]

    @property
    def model(self):
        model = xgb.XGBRegressor(**self.params)
        model.load_model(self.model_fpath)
        return model

    @property
    def cv_models(self):
        main_model_fpath = Path(self.model_fpath)
        model_fpaths = list(Path(main_model_fpath.parent, "cv-estimators").glob("*"))

        for fpath in model_fpaths:
            model = xgb.XGBRegressor(**self.params)
            model.load_model(str(fpath))
            yield model

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

    def plot_observed_vs_predicted(self, log: bool = False):
        """Plot observed vs. predicted values."""
        pred = self.model.predict(self.X_test)
        obs = np.squeeze(self.y_test.to_numpy())

        # plot the observed vs. predicted values using seaborn
        sns.set_theme()
        sns.set_style("whitegrid")
        sns.set_context("paper")

        _, ax = plt.subplots()
        p1 = min(min(pred), min(obs))
        p2 = max(max(pred), max(obs))
        if log:
            ax.loglog([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
        else:
            ax.plot([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
        ax.scatter(pred, obs, alpha=0.3)

        # Fit a regression line for observed vs. predicted values, plot the regression
        # line so that it spans the entire plot, and print the correlation coefficient
        m, b = np.polyfit(pred, obs, 1)
        reg_line = [m * p1 + b, m * p2 + b]
        if log:
            ax.loglog([p1, p2], reg_line, color="red", lw=0.5)
        else:
            ax.plot([p1, p2], reg_line, color="red", lw=0.5)
        ax.text(
            0.05,
            0.95,
            f"r = {np.corrcoef(pred, obs)[0, 1]:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
        )

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

        if "imputed" in row["NaN strategy"] or "imputed" in row["collection"]:
            imputed = True
        else:
            imputed = False

        if row["collection"] is not None:
            fpath = Path(row["collection"])
            X = DataCollection.from_collection(fpath)

            if not imputed:
                fn_ext = fpath.suffix
                imp_stem = fpath.stem + "_imputed"
                imp_fn = Path(fpath.parent, imp_stem + fn_ext)
                X_imp = DataCollection.from_collection(imp_fn)

        else:
            X = DataCollection.from_ids([id for id in predictor_ids])

        Y = DataCollection.from_ids([id for id in rv_ids], y_band)
        Y.df = Y.df[["geometry", y_name]]

        Xy = MLCollection(X, Y)
        Xy.drop_NAs(verbose=1)

        if not imputed:
            Xy_imputed = MLCollection(X_imp, Y)
            Xy_imputed.drop_NAs(verbose=1)
        else:
            Xy_imputed = None

        stats = Stats(
            cv_nrmse=row["CV nRMSE"],
            cv_nrmse_std=row["CV nRMSE STD"],
            cv_r2=row["CV r-squared"],
            cv_r2_std=row["CV r-squared STD"],
            test_r2=row["Test r-squared"],
            predictor_importances=np.array(
                ast.literal_eval(row["Predictor importance"])
            ),
        )

        if isinstance(row["Filtered RV outliers"], str):
            filtered_y_outliers = ast.literal_eval(row["Filtered RV outliers"])
        else:
            filtered_y_outliers = []

        return cls(
            id=int(row.name.__str__()),  # type: ignore (pandas bug)
            run_id=row["Run ID"],
            y_name=y_name,
            Xy=Xy,
            Xy_imputed=Xy_imputed,
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


class Prediction:
    def __init__(
        self,
        trained_set: TrainedSet,
        new_data: Union[gpd.GeoDataFrame, pd.DataFrame],
        new_data_imputed: Optional[gpd.GeoDataFrame] = None,
    ):
        self.trained_set = trained_set
        self.new_data = new_data
        self.new_data_imputed = new_data_imputed

    @cached_property
    def df(self) -> gpd.GeoDataFrame:
        prediction = self.trained_set.model.predict(
            self.new_data[self.trained_set.Xy.X.cols].to_numpy()
        )
        d = {self.trained_set.y_name: prediction}
        df = gpd.GeoDataFrame(
            d,
            geometry=self.new_data.geometry,
            crs=self.new_data.crs,
            index=self.new_data.index,
        )
        print("Calculating Area of Applicability...")
        df["DI"], df["AOA"] = self.aoa(threshold=0.95)
        df[f"{self.trained_set.y_name}_masked_aoa"] = df[self.trained_set.y_name].where(
            df["AOA"] == 1
        )
        df["CoV"] = self.cov()
        return df

    def aoa(self, threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Area of Applicability"""
        folds = [fold[1] for fold in list(self.trained_set.cv)]

        if self.trained_set.Xy_imputed is not None:
            X_train = self.trained_set.X_train_imputed
        else:
            X_train = self.trained_set.X_train

        if self.new_data_imputed is not None:
            new_data = self.new_data_imputed
        else:
            new_data = self.new_data

        DIs, AoA = aoa(
            new_df=new_data.copy(),
            training_df=X_train.copy(),
            weights=self.trained_set.stats.predictor_importances,
            thres=threshold,
            fold_indices=folds,
        )

        return DIs, AoA

    def cov(self) -> np.ndarray:
        """Coefficient of Variation"""
        preds = []
        for model in self.trained_set.cv_models:
            preds.append(
                model.predict(self.new_data[self.trained_set.Xy.X.cols].to_numpy())
            )

        all_preds = np.vstack(preds)
        cov = coefficient_of_variation(all_preds)
        return cov


def coefficient_of_variation(data):
    """Calculate coefficient of variation."""
    return np.std(data, axis=0) / np.mean(data, axis=0)
