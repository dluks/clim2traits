from __future__ import annotations

import glob
import os
from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import geopandas as gpd
import pandas as pd

from utils.data_retrieval import gdf_from_list
from utils.geodata import drop_XY_NAs, merge_gdfs


class Unit(Enum):
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
    TIF = "tif"


class CollectionName(Enum):
    INAT = "iNaturalist Traits"
    MODIS = "MOD09GA.061 Terra Surface Reflectance Daily Global 1km and 500m"
    SOIL = "ISRIC World Soil Information"
    WORLDCLIM = "WorldClim Bioclimatic Variables"


class DatasetInfo:
    def __init__(
        self,
        res: float = 0.5,
        unit: Unit = Unit.DEGREE,
        parent_dir: str = "",
        file_ext: FileExt = FileExt.TIF,
        collection_name: CollectionName = CollectionName.INAT,
        transform: str = None,
        bio_ids: list[str] = [],
    ):
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
    def fpaths(self) -> list[str]:
        """Filenames for the dataset based on the collection name"""

        if self.collection_name == CollectionName.INAT:
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


@dataclass
class Dataset:
    info: DatasetInfo
    description: str = None

    @cached_property
    def df(self) -> gpd.GeoDataFrame:
        return gdf_from_list(fns=self.info.fpaths)

    @cached_property
    def cols(self) -> pd.Index:
        return self.df.columns.difference(["geometry"])


@dataclass
class DataCollection:
    datasets: list[Dataset]

    @cached_property
    def df(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of all the datasets in the collection"""
        df = merge_gdfs([dataset.df for dataset in self.datasets])
        df = df.drop(columns=["x", "y", "band", "spatial_ref"])
        return df

    @cached_property
    def cols(self) -> pd.Index:
        return self.df.columns.difference(["geometry"])


class MLCollection:
    def __init__(self, X: DataCollection, Y: DataCollection):
        self.X = X
        self.Y = Y

    @cached_property
    def df(self) -> gpd.GeoDataFrame:
        """Returns a GeoDataFrame of all the datasets in the collection"""
        return self.X.df.merge(self.Y.df, on="geometry")

    def drop_NAs(self, verbose=0) -> None:
        """Drop NAs from the collection"""
        self.df, self.X.cols, self.Y.cols = drop_XY_NAs(
            self.df, self.X.cols, self.Y.cols, verbose=verbose
        )
        self.X.df = self.X.df[["geometry", *self.X.cols]]
        self.Y.df = self.Y.df[["geometry", *self.Y.cols]]
