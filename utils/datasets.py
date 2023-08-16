from __future__ import annotations

import glob
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as riox  # type: ignore
import xarray as xr

from utils.data_retrieval import gdf_from_list
from utils.dataset_tools import FileExt, Unit
from utils.geodata import drop_XY_NAs, get_epsg, mask_oceans, merge_dfs
from utils.training import TrainingConfig, TrainingRun


class CollectionName(Enum):
    """Name of the dataset collection."""

    def __init__(
        self,
        value,
        short,
        abbr,
        parent_dir,
        feature_key: Optional[Union[str, list[str]]] = None,
    ):
        self._value_ = value
        self.short = short
        self.abbr = abbr
        self.parent_dir = parent_dir
        self.feature_key = feature_key

    def __new__(
        cls,
        value,
        short,
        abbr,
        parent_dir,
        feature_key: Optional[Union[str, list[str]]] = None,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.short = short
        obj.abbr = abbr
        obj.parent_dir = parent_dir
        obj.feature_key = feature_key
        return obj

    INAT = (
        "iNaturalist Traits",
        "iNat_orig",
        "inat_orig",
        Path("./iNaturalist_traits/maps_iNaturalist"),
    )
    INAT_DGVM = (
        "iNaturalist Traits (DGVM)",
        "iNat_DGVM",
        "inat_dgvm",
        Path("./iNaturalist_traits/maps_iNaturalist/DGVM/continuous_traits"),
    )
    INAT_GBIF = (
        "GBIF Traits (TRY Gap-Filled)",
        "iNat_GBIF",
        "inat_gbif",
        Path("./iNaturalist_traits/maps_iNaturalist/GBIF/continuous_traits"),
    )
    INAT_SPLOT = (
        "sPlot Traits (TRY Non-Gap-Filled)",
        "iNat_sPlot",
        "inat_splot",
        Path("./iNaturalist_traits/maps_sPlotOpen"),
    )
    INAT_SPLOT_GF = (
        "sPlot Traits (TRY Gap-Filled)",
        "sPlot_GF",
        "inat_splot_gf",
        Path("./iNaturalist_traits/maps_iNaturalist/GBIF/continuous_traits"),
    )
    GBIF = (
        "GBIF Traits (TRY Gap-Filled, outliers removed)",
        "GBIF",
        "gbif",
        Path("./GBIF_trait_maps/global_maps/Shrub_Tree_Grass"),
    )
    SPLOT = (
        "sPlotOpen Traits (TRY Gap-Filled)",
        "sPlotOpen",
        "splot",
        Path("./GBIF_trait_maps/global_maps/Shrub_Tree_Grass"),
    )
    MODIS = (
        "MOD09GA.061 Terra Surface Reflectance Daily Global 1km and 500m",
        "MOD09GA.061",
        "modis",
        Path("./data/modis"),
        "sur_refl",
    )
    SOIL = (
        "ISRIC World Soil Information",
        "ISRIC_soil",
        "soil",
        Path("./data/soil"),
        ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"],
    )
    WORLDCLIM = (
        "WorldClim Bioclimatic Variables",
        "WC_BIO",
        "wc",
        Path("./data/worldclim/bio"),
        "wc2.1",
    )
    VODCA = "VODCA", "VODCA", "vodca", Path("./data/vodca"), "vodca"
    OTHER = "Other", "other", "other", Path("./data/other")

    @classmethod
    def from_short(cls, short: str) -> CollectionName:
        """Return the CollectionName enum value from the short name."""
        for name in cls:
            if name.short == short:
                return name
        raise ValueError(f"Invalid short name: {short}")


class GBIFBand(Enum):
    """Band names for GBIF data."""

    def __init__(self, band_id, readable):
        self._value_ = band_id
        self.readable = readable

    def __new__(cls, band_id, readable):
        obj = object.__new__(cls)
        obj._value_ = band_id
        obj.readable = readable
        return obj

    COUNT = 1, "obs_count"
    MEAN = 2, "mean"
    MEDIAN = 3, "median"
    STD = 4, "std"
    Q05 = 5, "q05"
    Q95 = 6, "q95"

    @classmethod
    def from_readable(cls, readable: str) -> GBIFBand:
        """Return the GBIFBand enum value from the readable name."""
        for band in cls:
            if band.readable == readable:
                return band
        raise ValueError(f"Invalid readable name: {readable}")


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
        file_ext: FileExt = FileExt.ANY,
        collection_name: CollectionName = CollectionName.OTHER,
        band: Optional[GBIFBand] = None,
        transform: str = "",
        bio_ids: list[str] = ["1", "4", "7", "12", "13-14", "15"],
        filter_outliers: list = [],
        _fpaths: list[Path] = [],
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
        self.file_ext = file_ext
        self.collection_name = collection_name
        self.band = band
        self.transform = transform
        self.bio_ids = bio_ids
        self.filter_outliers = filter_outliers

        self.parent_dir = self.collection_name.parent_dir
        self._fpaths = _fpaths

        if _fpaths:
            # Set the file extension according to that of the first file found (this could
            # be an issue if multiple file extensions are present)
            self.file_ext = FileExt(_fpaths[0].suffix[1:])

        # Transform is required for INAT datasets
        if (
            self.collection_name == CollectionName.INAT
            or self.collection_name == CollectionName.INAT_DGVM
            or self.collection_name == CollectionName.SPLOT
        ) and not self.transform:
            self.transform = "exp_ln"

    def __str__(self):
        return self.id

    def __repr__(self):
        return self.id

    @cached_property
    def res_str(self) -> str:
        """Returns the resolution identifier as used in the dataset."""
        return f"{self.res:g}_{self.unit.abbr}"

    @cached_property
    def id(self) -> str:
        """Returns the dataset identifier."""
        band_str = f"_{self.band.readable}" if self.band else ""
        return f"{self.collection_name.short}{band_str}_{self.res_str}"

    @cached_property
    def _search_str(self) -> str:
        """Returns the search string for the dataset."""
        search_str = ""
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
        elif self.collection_name == CollectionName.INAT_GBIF:
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                f"GBIF*.{self.file_ext.value}",
            )
        elif self.collection_name == CollectionName.INAT_SPLOT_GF:
            search_str = os.path.join(
                self.parent_dir, self.res_str, f"sPlot*.{self.file_ext.value}"
            )
        elif (
            self.collection_name == CollectionName.GBIF
            or self.collection_name == CollectionName.SPLOT
        ):
            prefix = "GBIF" if self.collection_name == CollectionName.GBIF else "sPlot"
            if self.file_ext == FileExt.ANY:
                # Set file extension to .grd for GBIF and sPlot data
                # TODO: This is a hacky solution. Should probably rethink file ext
                # property assignment altogether.
                self.file_ext = FileExt.GRID
            search_str = os.path.join(
                self.parent_dir,
                self.res_str,
                f"{prefix}*.{self.file_ext.value}",
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

    @cached_property
    def fpaths(self) -> list[str]:
        """Filenames for the dataset based on the collection name"""
        if self._fpaths:
            return [str(fpath) for fpath in self._fpaths]
        if self.collection_name == CollectionName.WORLDCLIM:
            all_fpaths = self._get_fpaths()

            if not self.bio_ids:
                return all_fpaths

            fnames = []
            for bio_id in self.bio_ids:
                first_part = f"wc2.1_{self.res_str}"
                second_part = f"bio_{bio_id}"
                fname = f"{first_part}_{second_part}.tif"
                fnames.append(fname)

            fpaths = []
            for fname in fnames:
                for fpath in all_fpaths:
                    if fname in fpath:
                        fpaths.append(fpath)

            if not fpaths:
                raise FileNotFoundError("No files found!")

            # Set the file extension according to that of the first file found (this could
            # be an issue if multiple file extensions are present)
            self.file_ext = FileExt(Path(fpaths[0]).suffix[1:])
            return sorted(fpaths)

        return self._get_fpaths()

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

        # Set the file extension according to that of the first file found (this could
        # be an issue if multiple file extensions are present)
        self.file_ext = FileExt(Path(fpaths[0]).suffix[1:])
        return fpaths

    @cached_property
    def epsg(self) -> int:
        return get_epsg(self.fpaths[0])

    @cached_property
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Returns a dataframe of the dataset."""
        if self.collection_name == CollectionName.VODCA:
            ds_name = self.collection_name.abbr
        else:
            ds_name = None

        if (
            self.collection_name == CollectionName.GBIF
            or self.collection_name == CollectionName.SPLOT
        ) and not self.band:
            raise ValueError("Band must be specified for GBIF and sPlot data.")

        print(f"Reading GDFs from fpaths for {self.collection_name.short}...")
        df = gdf_from_list(
            fns=self.fpaths,
            ds_name=ds_name,
            band=self.band.value if self.band else None,
        )

        print("Dropping unnecessary columns...")
        df = df.drop(columns=["x", "y", "band", "spatial_ref"], errors="ignore")

        print("Masking oceans...")
        df = mask_oceans(df)  # Mask non-land points

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

    @classmethod
    def from_id(cls, id: str, band: Optional[str] = None) -> Dataset:
        """Returns a Dataset object from an identifier.

        Args:
            id (str): Dataset identifier.

        Returns:
            Dataset: Dataset object.
        """
        res_str = "_".join(id.split("_")[-2:])
        short = id.split(f"_{res_str}")[0]
        if band:
            short = short.replace(f"_{band}", "")
        collection_name = CollectionName.from_short(short)
        res = float(res_str.split("_")[0])
        unit = Unit.from_abbr(res_str.split("_")[1])
        ds_band = GBIFBand.from_readable(band) if band else None

        return cls(res=res, unit=unit, collection_name=collection_name, band=ds_band)

    @classmethod
    def from_stem(cls, stem=str) -> Dataset:
        """Returns a Dataset object from a filename stem.
        Note: the returned Dataset will not have an accurate unit, resolution, or collection_name.

        Args:
            stem (str): Filename stem.

        Returns:
            Dataset: Dataset object.
        """
        file_exts = ["tif", "nc", "grd"]  # Only consider these file extensions
        fpath = None
        for file_ext in file_exts:
            fpath = list(Path(".").glob(f"**/{stem}.{file_ext}"))
            if fpath:
                break
        if not fpath:
            raise FileNotFoundError(f"No files found for {stem}.")
        return cls(
            _fpaths=fpath,
        )


def resample_dataset(
    dataset: Dataset,
    resolution: Union[float, int],
    unit: Unit,
    format: str = "netcdf",
    mf: bool = False,
    chunks: Union[dict, str] = "auto",
) -> None:
    """Resamples a dataset to a new resolution and unit.

    Args:
        dataset (Dataset): Dataset to resample.
        resolution (Union[float, int]): New resolution.
        unit (Unit): New unit.
        format (str, optional): Format of the output file. Options are ["GTiff",
            "netcdf", "zarr"]. Defaults to "netcdf".
        mf (bool, optional): Whether to open files as a multi-file dataset. Defaults to
            False.
        chunks (Union[dict, str], optional): Chunks for the output dataset. Defaults to
            "auto".
    """
    # Check if the dataset resolution is the same as the new resolution
    if dataset.res == resolution and dataset.unit == unit:
        raise ValueError(f"Dataset resolution is already {resolution} {unit.abbr}")

    # Create the new directory if it doesn't exist
    new_dir = Path(dataset.parent_dir, f"{resolution}_{unit.abbr}")
    new_dir.mkdir(parents=True, exist_ok=True)

    res = resolution
    if unit == Unit.SECOND:
        res = resolution / 3600
    elif unit == Unit.KILOMETER:
        res = resolution * 0.00898315284120171538

    if format == "GTiff":
        ext = ".tif"
    elif format == "netcdf":
        ext = ".nc"
    elif format == "zarr":
        ext = ".zarr"

    if mf:
        ds = xr.open_mfdataset(dataset.fpaths, parallel=True, chunks=chunks)
        ds = reproject_dataset(ds, resolution, dataset.id)
        fpath = new_dir / f"{dataset.id}{ext}"
        write_dataset(ds, fpath, format)

        print("Reprojecting...")
        ds = reproject_dataset(ds, res, ds_name)

        print("Writing...")
        # Write the dataset (necessary because clipping can only utilize chunks if
        # it is first written to and then read from disk)
        write_dataset(ds, fpath, format)
        ds.close()

        print("Clipping...")
        # Load the dataset and clip
        ds = xr.open_dataset(fpath, chunks=chunks)
        ds = clip_and_pad_dataset(ds)

        print("Writing again...")
        write_dataset(ds, fpath, format)
    else:
        for fpath in dataset.fpaths[:1]:
            fpath = Path(fpath)
            new_fname = fpath.name.replace(dataset.res_str, f"{resolution}_{unit.abbr}")

            if dataset.collection_name == CollectionName.SOIL:
                # Append the soil variable subdirectory
                soil_var_dir = new_dir / fpath.parts[-2]
                soil_var_dir.mkdir(parents=True, exist_ok=True)
                new_fname = Path(soil_var_dir, new_fname)

            # Create the new filename
            new_fpath = new_dir / new_fname
            new_fpath = new_fpath.with_suffix(ext)

            ds_name = new_fpath.stem

            print(str(new_fpath))

            # # Open the dataset
            # if fpath.suffix == ".tif":
            #     ds = riox.open_rasterio(fpath, masked=True, chunks=chunks, lock=False)
            # else:
            #     ds = xr.open_dataset(fpath, chunks=chunks)

            # print("Reprojecting...")
            # ds = reproject_dataset(ds, res, ds_name)

            # print("Writing...")
            # # Write the dataset (necessary because clipping can only utilize chunks if
            # # it is first written to and then read from disk)
            # write_dataset(ds, new_fpath, format)
            # ds.close()

            print("Clipping...")
            # Load the dataset and clip
            ds = xr.open_dataset(new_fpath, engine="netcdf4", chunks=chunks)
            ds = clip_and_pad_dataset(ds)
            # ds.close()
            # add "_clipped" to the path
            new_fpath = new_fpath.with_name(
                f"{new_fpath.stem}_clipped{new_fpath.suffix}"
            )
            print("Writing again...")
            write_dataset(ds, new_fpath, format)
            # ds = resample_gdal(
            #     in_fn=str(fpath),
            #     out_fn=str(new_fpath),
            #     res=res,
            #     epsg=f"EPSG:4326",
            #     globe=True,
            #     format=format,
            # )

    ds.close()
    del ds


def reproject_dataset(ds: xr.Dataset, res: float, ds_name: str) -> xr.Dataset:
    # set CRS to EPSG: 4326 if it doesn't exist
    if not ds.rio.crs:
        ds.rio.write_crs("EPSG:4326", inplace=True)

    # Make an empty dataset with the new extent and resolution
    # Define the global extent
    xmin, ymin, xmax, ymax = (-180.0, -90.0, 180.0, 90.0)

    # Calculate the number of rows and columns based on the resolution
    nrows = int((ymax - ymin) / res)
    ncols = int((xmax - xmin) / res)

    # Coordinates represent the center of the grid cells, so we need to adjust the
    # coordinates by half the resolution
    shift = res / 2
    y = np.linspace(ymax - shift, ymin + shift, nrows)
    x = np.linspace(xmin + shift, xmax - shift, ncols)

    # Create the empty DataArray with the desired dimensions and fill value
    target_grid = xr.DataArray(
        data=np.ones((nrows, ncols)),
        dims=("y", "x"),
        coords={
            "y": y,
            "x": x,
        },
    )

    target_grid.rio.write_crs("EPSG:4326", inplace=True)  # Set the CRS

    # Reproject to the new CRS and resolution
    # First rename lat and lon to y and x if they exist in coords
    if "lat" in ds.coords:
        ds = ds.rename({"lat": "y"})
    if "lon" in ds.coords:
        ds = ds.rename({"lon": "x"})

    # ds = ds.rio.reproject_match(match_data_array=target_grid, resampling=4)
    ds = ds.rio.reproject(
        dst_crs="EPSG:4326", resolution=res, resampling=4, num_threads=15
    )

    # Replace the default variable name with the dataset name
    if isinstance(ds, xr.DataArray):
        if ds.name is None:
            ds = ds.rename(ds_name)

    return ds


def clip_and_pad_dataset(ds: xr.Dataset) -> xr.Dataset:
    land_mask = gpd.read_feather("./data/masks/land_mask_110m.feather")
    if not ds.rio.crs:
        ds.rio.write_crs("EPSG:4326", inplace=True)
    df = dask_geopandas
    ds = ds.rio.clip(
        geometries=land_mask.geometry.values,
        crs=land_mask.crs,
        drop=False,
        invert=False,
        from_disk=False,
    )

    ds = ds.rio.pad_box(minx=180, miny=-90, maxx=180, maxy=90)
    return ds


def write_dataset(ds: xr.Dataset, fpath: Union[str, os.PathLike], format: str) -> Path:
    if format == "GTiff":
        ds.rio.to_raster(
            fpath,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
        )
    else:
        # Drop "grid_mapping" attr from data variables (netcdf can't handle this)
        for var in ds.data_vars:
            if "grid_mapping" in ds[var].attrs:
                ds[var].attrs.pop("grid_mapping")
        ds.close()
        ds.to_netcdf(fpath, mode="w", engine="netcdf4", compute=False)


@dataclass
class DataCollection:
    """Represents a collection of datasets.

    Attributes:
        datasets (list[Dataset]): List of Dataset objects in the collection.
        collection_file (Optional[Union[str, os.PathLike]]): Path to a collection file.

    Methods:
        df() -> gpd.GeoDataFrame:
            Returns a GeoDataFrame of all the datasets in the collection.
        cols() -> pd.Index:
            Returns the columns of the collection.
    """

    datasets: list[Dataset]
    collection_file: Optional[Union[str, os.PathLike]] = None

    @cached_property
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
        """Returns a GeoDataFrame of all the datasets in the collection"""
        df = merge_dfs([dataset.df for dataset in self.datasets])
        return df

    @cached_property
    def cols(self) -> Union[pd.Index, str]:
        return self.df.columns.difference(["geometry"])

    @classmethod
    def from_ids(cls, ids: list[str], band: Optional[str] = None) -> DataCollection:
        """Returns a DataCollection object from a list of identifiers.

        Args:
            ids (list[str]): List of dataset identifiers.

        Returns:
            DataCollection: DataCollection object.
        """
        datasets = [Dataset.from_id(id, band=band) for id in ids]
        return cls(datasets=datasets)

    @classmethod
    def from_df(cls, df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> DataCollection:
        """Returns a DataCollection object from a DataFrame.

        Args:
            df (Union[pd.DataFrame, gpd.GeoDataFrame]): DataFrame.

        Returns:
            DataCollection: DataCollection object.
        """
        # Determine the datasets by checking to see if CollectionName feature_keys are
        # present in the DataFrame columns
        dataset_ids = dataset_ids_from_df(df)
        if not dataset_ids:
            raise ValueError("Cannot find any dataset identifiers in the dataframe.")

        res_str = check_for_res_str(df)
        if not res_str:
            raise ValueError("Cannot find a valid resolution string in the dataframe.")

        # Update the dataset_ids with the res_str
        dataset_ids = [f"{id}_{res_str}" for id in dataset_ids]

        # Create the DataCollection
        data_collection = cls.from_ids(dataset_ids)

        # Overwrite the dataframe with the preexisting one
        # (TODO: For an imputed DataCollection with non-imputed single datasets, this
        # means that the DataCollection dataframe may contain different values than are
        # present in the individual datasets. This is not ideal.)
        data_collection.df = df

        return data_collection

    @classmethod
    def from_collection(cls, collection: Union[str, os.PathLike]) -> DataCollection:
        df = gpd.read_feather(collection)
        data_collection = cls.from_df(df)
        data_collection.collection_file = str(collection)
        return data_collection


def dataset_ids_from_df(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> list[str]:
    # Determine the datasets by checking to see if CollectionName feature_keys are
    # present in the DataFrame columns
    dataset_ids = []
    for collection_name in CollectionName:
        feature_key = collection_name.feature_key
        if isinstance(feature_key, str):
            if any(feature_key in col for col in df.columns):
                dataset_ids.append(collection_name.short)
        elif isinstance(feature_key, list):
            if all(any(key in col for col in df.columns) for key in feature_key):
                dataset_ids.append(collection_name.short)
    return dataset_ids


def check_for_res_str(df: Union[pd.DataFrame, gpd.GeoDataFrame]) -> Union[str, None]:
    """Checks a DataFrame for a resolution string.
    Returns the resolution string if found, otherwise returns None.
    """
    # Get a res_str from the DataFrame by checking the column names
    common_res_strings = ["0.5_deg", "1_deg", "2_deg", "1_km", "1000_m"]
    for col in df.columns:
        for res_str in common_res_strings:
            if res_str in col:
                return res_str
    return None


class MLCollection:
    """Represents a collection of datasets for machine learning.

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
    def df(self) -> Union[pd.DataFrame, gpd.GeoDataFrame, pd.Series]:
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
        tune_params: bool = False,
        resume: bool = False,
        debug: bool = False,
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
            if isinstance(y_cols, pd.Index):
                y_cols = y_cols[y_idx]
            else:
                warnings.warn("Ignoring y_idx because Y.cols is a string.")

        if resume:
            results_csv = pd.read_csv(training_config.results_csv)
            run_ids = results_csv["Run ID"]
            run_id = run_ids.iloc[run_ids.last_valid_index()]  # type: ignore
            runs = results_csv.loc[results_csv["Run ID"] == run_id]
            completed_rvs = runs["Response variable"].values
            if isinstance(y_cols, pd.Index):
                y_cols = y_cols.difference(completed_rvs)
            elif y_cols in completed_rvs:
                raise ValueError("Response variable provided already completed.")

        for i, y_col in enumerate(y_cols):
            if not resume:
                # Only resume after first response var in the collection has completed
                resume = False if i == 0 else True

            print(f"Training model on {y_col}...")

            train_run = TrainingRun(self, y_col, training_config, resume=resume)

            if tune_params:
                print("Tuning hyperparameters...")
                train_run.tune_params_cv()
                print("Tuning complete.")
            else:
                train_run.train_cv()

            print("Training model on all data...")
            train_run.train_model_on_all_data()
            print("Training complete.")

            print("Saving results...")
            train_run.save_results()
            print("Results saved.")
