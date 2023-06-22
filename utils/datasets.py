import glob
import os
import pathlib
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Union

import geopandas as gpd
import pandas as pd

from utils.data_retrieval import gdf_from_list
from utils.gdal import resample_gdal
from utils.geodata import drop_XY_NAs, get_epsg, merge_dfs
from utils.training import TrainingConfig, TrainingRun


class Unit(Enum):
    """Unit of measurement for the dataset."""

    def __new__(cls, value, abbr):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.abbr = abbr
        return obj

    DEGREE = "degree", "deg"
    METER = "meter", "m"
    KILOMETER = "kilometer", "km"
    MINUTE = "minute", "min"


class FileExt(Enum):
    """File extension for the raw data in the dataset."""

    TIF = "tif"


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
    MODIS = (
        "MOD09GA.061 Terra Surface Reflectance Daily Global 1km and 500m",
        "MOD09GA.061_1km",
        "modis",
    )
    SOIL = "ISRIC World Soil Information", "ISRIC_soil", "soil"
    WORLDCLIM = "WorldClim Bioclimatic Variables", "WC_BIO", "wc"
    OTHER = "Other", "other", "other"


class Dataset:
    """
    Represents a dataset with specific attributes such as name, resolution, unit of
    measurement, and file extension.

    Attributes:
        name (str): Name of the dataset.
        res (float): Resolution of the dataset.
        unit (Unit): Unit of measurement for the dataset.
        parent_dir (pathlib.Path): Parent directory of the dataset.
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
        unit: Unit = Unit.DEGREE,
        parent_dir: pathlib.Path = pathlib.Path.cwd(),
        file_ext: FileExt = FileExt.TIF,
        collection_name: CollectionName = CollectionName.OTHER,
        transform: str = "",
        bio_ids: list[str] = [],
    ):
        """Initialize a Dataset object with resolution, unit, and collection name.

        Args:
            name (str, optional): Name of the dataset. Defaults to "".
            res (float, optional): Resolution of the dataset. Defaults to 0.5.
            unit (Unit, optional): Unit of measurement for the dataset. Defaults to
                Unit.DEGREE.
            parent_dir (pathlib.Path, optional): Parent directory of the dataset.
                Defaults to pathlib.Path.cwd().
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

        # Transform is required for INAT datasets
        if self.collection_name == CollectionName.INAT and not self.transform:
            self.transform = "ln"

    @property
    def res_str(self) -> str:
        """Returns the resolution identifier as used in the dataset."""
        return f"{str(self.res)}_{self.unit.abbr}"

    @property
    def id(self) -> str:
        """Returns the dataset identifier."""
        return f"{self.collection_name.short}_{self.res_str}"

    @property
    def fpaths(self) -> list[str]:
        """Filenames for the dataset based on the collection name"""

        if (
            self.collection_name == CollectionName.INAT
            or self.collection_name == CollectionName.INAT_DGVM
        ):
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                self.transform,
                f"*.{self.file_ext.value}",
            )
            return sorted(glob.glob(search_str))

        if self.collection_name == CollectionName.WORLDCLIM:
            search_str = os.path.join(
                self.parent_dir, self.res_str, f"*.{self.file_ext.value}"
            )
            all_fpaths = sorted(glob.glob(search_str))

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

            return sorted(fpaths)

        if self.collection_name == CollectionName.SOIL:
            search_str = os.path.join(
                self.parent_dir,
                "*",
                self.res_str,
                f"*.{self.file_ext.value}",
            )
            return sorted(glob.glob(search_str))

        if self.collection_name == CollectionName.MODIS:
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                f"*.{self.file_ext.value}",
            )
            return sorted(glob.glob(search_str))

        raise ValueError("No filepaths found!")

    @cached_property
    def epsg(self) -> int:
        return get_epsg(self.fpaths[0])

    @cached_property
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        df = gdf_from_list(fns=self.fpaths)
        df = df.drop(columns=["x", "y", "band", "spatial_ref"], errors="ignore")
        return df

    @cached_property
    def cols(self) -> pd.Index:
        return self.df.columns.difference(["geometry"])


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
    new_dir = pathlib.Path(dataset.parent_dir, f"{resolution}_{unit.abbr}")
    new_dir.mkdir(parents=True, exist_ok=True)

    for fpath in dataset.fpaths:
        fpath = pathlib.Path(fpath)
        new_fname = fpath.name.replace(dataset.res_str, f"{resolution}_{unit.abbr}")

        if dataset.collection_name == CollectionName.SOIL:
            # Append the soil variable subdirectory
            soil_var_dir = new_dir / fpath.parts[-2]
            soil_var_dir.mkdir(parents=True, exist_ok=True)
            new_fname = pathlib.Path(soil_var_dir, new_fname)

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
        df = df.drop(columns=["x", "y", "band", "spatial_ref"])
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
        y_idx: list[int] = None,
        resume: bool = False,
    ) -> None:
        """Train models for all response variables in MLCollection

        Args:
            training_config (TrainingConfig): Training configuration
            y_idx (list[int], optional): List of indices of response variables to train.
                Defaults to None.
            resume (bool, optional): Whether to resume training. Defaults to False.
        """
        cols = self.Y.cols

        if y_idx:
            cols = cols[y_idx]

        for i, y_col in enumerate(cols):
            if not resume:
                resume = (
                    False if i == 0 else True
                )  # Only resume if not first response var in the collection

            train_run = TrainingRun(self, y_col, training_config, resume=resume)

            train_run.tune_params_cv()
            train_run.train_model_on_all_data()
            train_run.save_results()
            self.training_runs.append(train_run)
