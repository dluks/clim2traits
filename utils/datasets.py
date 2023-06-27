from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import geopandas as gpd
import numpy as np
import pandas as pd

from utils.data_retrieval import gdf_from_list
from utils.dataset_tools import FileExt, Unit
from utils.gdal import resample_gdal
from utils.geodata import drop_XY_NAs, get_epsg, merge_dfs
from utils.training import TrainingConfig, TrainingRun


class CollectionName(Enum):
    """Name of the dataset collection."""

    def __new__(cls, value, short, abbr):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.short = short
        obj.abbr = abbr
        return obj

    INAT = "iNaturalist Traits", "iNat_orig", "inat_orig"
    INAT_DGVM = "iNaturalist Traits (DGVM)", "iNat_DGVM", "inat_dgvm"
    INAT_GBIF = "iNaturalist Traits (GBIF)", "iNat_GBIF", "inat_gbif"
    SPLOT = "sPlot Open Traits", "sPlotOpen", "splot"
    MODIS = (
        "MOD09GA.061 Terra Surface Reflectance Daily Global 1km and 500m",
        "MOD09GA.061_1km",
        "modis",
    )
    SOIL = "ISRIC World Soil Information", "ISRIC_soil", "soil"
    WORLDCLIM = "WorldClim Bioclimatic Variables", "WC_BIO", "wc"
    VODCA = "VODCA", "VODCA", "vodca"
    OTHER = "Other", "other", "other"


class Dataset:
    """
    Represents a dataset with specific attributes such as name, resolution, unit of
    measurement, and file extension.

    Attributes:
        name (str): Name of the dataset.
        res (float): Resolution of the dataset.
        unit (Unit): Unit of measurement for the dataset.
        parent_dir (Path): Parent directory of the dataset.
        file_ext (FileExt): File extension for the raw data in the dataset.
        collection_name (CollectionName): Name of the dataset collection.
        transform (str): Transformation type of the dataset (applies to iNaturalist
            data).
        bio_ids (list[str]): List of bioclimatic variable IDs (applies to WorldClim
            data).

    Methods:
        res_str() -> str:
            Returns the resolution identifier as used in the dataset.
        id() -> str:
            Returns the dataset identifier.
        fpaths() -> list[str]:
            Returns filenames for the dataset based on the collection name.
        df() -> gpd.GeoDataFrame:
            Returns a GeoDataFrame of the dataset.
        cols() -> pd.Index:
            Returns the columns of the dataset.
    """

    def __init__(
        self,
        name: str = "",
        res: Union[float, int] = 0.5,
        unit: Unit = Unit("degree"),
        parent_dir: Path = Path.cwd(),
        file_ext: FileExt = FileExt("tif"),
        collection_name: CollectionName = CollectionName.OTHER,
        transform: str = "",
        bio_ids: list[str] = [],
        filter_outliers: list = [],
    ):
        """Initialize a Dataset object with resolution, unit, and collection name.

        Args:
            name (str, optional): Name of the dataset. Defaults to "".
            res (float, optional): Resolution of the dataset. Defaults to 0.5.
            unit (Unit, optional): Unit of measurement for the dataset. Defaults to
                Unit.DEGREE.
            parent_dir (Path, optional): Parent directory of the dataset.
                Defaults to Path.cwd().
            file_ext (FileExt, optional): File extension for the raw data in the
                dataset. Defaults to FileExt.TIF.
            collection_name (CollectionName, optional): Name of the dataset collection.
                Defaults to CollectionName.OTHER.
            transform (str, optional): Transformation type of the dataset. Only applies
                to iNaturalist data. Defaults to "".
            bio_ids (list[str], optional): List of bioclimatic variable IDs. Only
                applies to WorldClim data. If empty then all vairables will be used.
                Defaults to [].
        """
        self.name = name
        self.res = res
        self.unit = unit
        self.parent_dir = parent_dir
        self.file_ext = file_ext
        self.collection_name = collection_name
        self.transform = transform
        self.bio_ids = bio_ids
        self.filter_outliers = filter_outliers

        # Transform is required for INAT datasets
        if (
            self.collection_name == CollectionName.INAT
            or self.collection_name == CollectionName.INAT_DGVM
            or self.collection_name == CollectionName.SPLOT
        ) and not self.transform:
            self.transform = "exp_ln"

    @property
    def res_str(self) -> str:
        """Returns the resolution identifier as used in the dataset."""
        return f"{self.res:g}_{self.unit.abbr}"

    @property
    def id(self) -> str:
        """Returns the dataset identifier."""
        return f"{self.collection_name.short}_{self.res_str}"

    @property
    def _search_str(self) -> str:
        """Returns the search string for the dataset."""
        search_str = ""
        if (
            self.collection_name == CollectionName.INAT
            or self.collection_name == CollectionName.INAT_DGVM
            or self.collection_name == CollectionName.SPLOT
        ):
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                self.transform,
                f"*.{self.file_ext.value}",
            )
        elif self.collection_name == CollectionName.INAT_GBIF:
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                f"GBIF*.{self.file_ext.value}",
            )
        elif self.collection_name == CollectionName.SOIL:
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                "*",
                f"*.{self.file_ext.value}",
            )
        else:
            search_str = os.path.join(
                self.parent_dir, self.res_str, f"*.{self.file_ext.value}"
            )
        return search_str

    @property
    def fpaths(self) -> list[str]:
        """Filenames for the dataset based on the collection name"""

        if self.collection_name == CollectionName.WORLDCLIM:
            all_fpaths = self._get_fpaths()

            if not self.bio_ids:
                return all_fpaths

            fnames = []
            for bio_id in self.bio_ids:
                first_part = f"wc2.1_{self.res_str}"
                second_part = f"bio_{bio_id}"
                fname = f"{first_part}_{second_part}.{self.file_ext.value}"
                fnames.append(fname)

            fpaths = []
            for fname in fnames:
                for fpath in all_fpaths:
                    if fname in fpath:
                        fpaths.append(fpath)

            if not fpaths:
                raise FileNotFoundError("No files found!")

            return sorted(fpaths)

        return self._get_fpaths()

    @cached_property
    def epsg(self) -> int:
        return get_epsg(self.fpaths[0])

    @cached_property
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        if self.collection_name == CollectionName.VODCA:
            ds_name = self.collection_name.abbr
        else:
            ds_name = None
        df = gdf_from_list(fns=self.fpaths, ds_name=ds_name)
        df = df.drop(columns=["x", "y", "band", "spatial_ref"], errors="ignore")
        if self.filter_outliers:
            df = self._filter_outliers(df, self.filter_outliers)
        return df

    @staticmethod
    def _filter_outliers(
        df: Union[pd.DataFrame, gpd.GeoDataFrame], bounds: list = [0.01, 0.99]
    ) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Filters outliers from a dataset.

        Args:
            df (Union[pd.DataFrame, gpd.GeoDataFrame]): Dataset.
            bounds (list, optional): Lower and upper bounds for filtering outliers.

        Returns:
            Union[pd.DataFrame, gpd.GeoDataFrame]: Dataset with outliers removed.
        """
        # Filter out outliers
        cols = df.columns.difference(["geometry"]).values

        for col in cols:
            lower_mask = df[col] < df[col].quantile(bounds[0])
            upper_mask = df[col] > df[col].quantile(bounds[1])

            # combine the masks and set the values to NaN
            df[col] = df[col].mask(lower_mask | upper_mask)

        return df

    @cached_property
    def cols(self) -> pd.Index:
        return self.df.columns.difference(["geometry"])

    def _get_fpaths(self) -> list[str]:
        """Returns a list of filepaths for a given resolution string.

        Returns:
            list[str]: List of filepaths.
        """
        fpaths = sorted(glob.glob(self._search_str))

        if not fpaths:
            # Check for variations in the resolution string (e.g. 0.5deg vs 0.5_deg)
            variations = [
                self.res_str,
                self.res_str.replace(".", ""),
                self.res_str.replace("_", ""),
                self.res_str.replace("_", "").replace(".", ""),
            ]

            for variation in variations:
                fpaths = sorted(
                    glob.glob(self._search_str.replace(self.res_str, variation))
                )
                if fpaths:
                    break

        if not fpaths:
            raise FileNotFoundError(f"No files found for {self.collection_name}.")

        return fpaths


def resample_dataset(
    dataset: Dataset, resolution: Union[float, int], unit: Unit
) -> None:
    """Resamples a dataset to a new resolution and unit.

    Args:
        dataset (Dataset): Dataset to resample.
        resolution (Union[float, int]): New resolution.
        unit (Unit): New unit.
    """
    # Check if the dataset resolution is the same as the new resolution
    if dataset.res == resolution and dataset.unit == unit:
        raise ValueError(f"Dataset resolution is already {resolution} {unit.abbr}")

    # Create the new directory if it doesn't exist
    new_dir = Path(dataset.parent_dir, f"{resolution}_{unit.abbr}")
    new_dir.mkdir(parents=True, exist_ok=True)

    for fpath in dataset.fpaths:
        fpath = Path(fpath)
        new_fname = fpath.name.replace(dataset.res_str, f"{resolution}_{unit.abbr}")

        if dataset.collection_name == CollectionName.SOIL:
            # Append the soil variable subdirectory
            soil_var_dir = new_dir / fpath.parts[-2]
            soil_var_dir.mkdir(parents=True, exist_ok=True)
            new_fname = Path(soil_var_dir, new_fname)

        # Create the new filename
        new_fpath = new_dir / new_fname

        print(str(fpath))
        ds = resample_gdal(
            in_fn=str(fpath),
            out_fn=str(new_fpath),
            res=resolution,
            epsg=f"EPSG:{str(dataset.epsg)}",
            globe=True,
        )

        del ds


@dataclass
class DataCollection:
    """
    Represents a collection of datasets.

    Attributes:
        datasets (list[Dataset]): List of Dataset objects in the collection.

    Methods:
        df() -> gpd.GeoDataFrame:
            Returns a GeoDataFrame of all the datasets in the collection.
        cols() -> pd.Index:
            Returns the columns of the collection.
    """

    datasets: list[Dataset]

    @cached_property
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Returns a GeoDataFrame of all the datasets in the collection"""
        df = merge_dfs([dataset.df for dataset in self.datasets])
        return df

    @cached_property
    def cols(self) -> pd.Index:
        return self.df.columns.difference(["geometry"])


class MLCollection:
    """
    Represents a collection of datasets for machine learning.

    Attributes:
        X (DataCollection): DataCollection object for X.
        Y (DataCollection): DataCollection object for Y.
        training_runs (list[dict]): List of training runs.

    Methods:
        df() -> gpd.GeoDataFrame:
            Returns a GeoDataFrame of all the datasets in the collection.
        coords() -> pd.Series:
            Returns a Series of the coordinates of the collection.
        drop_NAs(verbose=0) -> None:
            Drops NAs from the collection.
        train_y_models(config: dict) -> None:
            Trains models for each column in Y.
    """

    def __init__(self, X: DataCollection, Y: DataCollection):
        """Initialize an MLCollection object with X and Y datasets or collections.

        Args:
            X (DataCollection): DataCollection object for X.
            Y (DataCollection): DataCollection object for Y.
        """
        self.X = X
        self.Y = Y
        self.training_runs = []

    @cached_property
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Returns a GeoDataFrame of all the datasets in the collection"""
        return self.X.df.merge(self.Y.df, on="geometry")

    @cached_property
    def coords(self) -> pd.Series:
        """Returns a Series of the coordinates of the collection"""
        return self.df.geometry

    def drop_NAs(self, verbose=0) -> None:
        """Drop NAs from the collection"""
        self.df, self.X.cols, self.Y.cols = drop_XY_NAs(
            self.df, self.X.cols, self.Y.cols, verbose=verbose
        )
        self.X.df = self.X.df[["geometry", *self.X.cols]]
        self.Y.df = self.Y.df[["geometry", *self.Y.cols]]

    def add_training_run(self, training_run: TrainingRun) -> None:
        self.training_runs.append(training_run)

    def train_Y_models(
        self,
        training_config: TrainingConfig,
        y_idx: Optional[list[int]] = None,
        resume: bool = False,
    ) -> None:
        """Train models for all response variables in MLCollection

        Args:
            training_config (TrainingConfig): Training configuration
            y_idx (list[int], optional): List of indices of response variables to train.
                Defaults to None.
            resume (bool, optional): Whether to resume training. Defaults to False.
        """
        y_cols = self.Y.cols

        if y_idx:
            y_cols = y_cols[y_idx]

        if resume:
            results_csv = pd.read_csv(training_config.results_csv)
            run_ids = results_csv["Run ID"]
            run_id = run_ids.iloc[run_ids.last_valid_index()]  # type: ignore
            runs = results_csv.loc[results_csv["Run ID"] == run_id]
            completed_rvs = runs["Response variable"].values
            y_cols = y_cols.difference(completed_rvs)

        for i, y_col in enumerate(y_cols):
            if not resume:
                # Only resume after first response var in the collection has completed
                resume = False if i == 0 else True

            print(f"Training model on {y_col}...")

            train_run = TrainingRun(self, y_col, training_config, resume=resume)

            print("Tuning hyperparameters...")
            train_run.tune_params_cv()
            print("Tuning complete.")
            print("Training model on all data...")
            train_run.train_model_on_all_data()
            print("Training complete.")
            print("Saving results...")
            train_run.save_results()
            print("Results saved.")
