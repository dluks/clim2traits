################################
# Geodata utils
################################
import os
from functools import reduce
from typing import Optional, Tuple, Union

import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling

NPARTITIONS = os.cpu_count()


def ds2gdf(ds: xr.DataArray) -> gpd.GeoDataFrame:
    """Converts an xarray dataset to a geopandas dataframe

    Args:
        ds (xarray.DataArray): Dataset to convert

    Returns:
        geopandas.GeoDataFrame: GeoPandas dataframe
    """
    df = ds.to_dask_dataframe().reset_index()
    geometry = dgpd.points_from_xy(df.x, df.y)
    df = dgpd.from_dask_dataframe(df, geometry=geometry)
    df = df.set_crs(ds.rio.crs)
    df = df.drop(columns=["band", "spatial_ref"])
    return df


def tif2gdf(
    raster: Union[str, xr.DataArray], band_id: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Converts a GeoTIFF to a GeoPandas data frame. Accepts either the filename of a
    GeoTiff or an xr dataset.

    Args:
        raster (str || xarray.DataArray): Filename of the GeoTIFF raster or the opened
            raster itself.
        band (Optional[int], optional): Band number. Defaults to None.

    Returns:
        geopandas.GeoDataFrame: GeoPandas data frame
    """
    ds = validate_raster(raster)
    band_gdfs = []

    if band_id:
        if "long_name" in ds.attrs.keys():
            band_name = ds.attrs["long_name"][band_id - 1]  # Band ints aren't 0-indexed
        else:
            band_name = f"_band{band_id:02d}"
        band_ds = ds.sel(band=band_id)
        band_ds.name = f"{ds.name}_{band_name}"
        band_gdfs.append(ds2gdf(band_ds))
    else:
        for band in ds.band.values:
            band_ds = ds.sel(band=band)
            if ds.band.values.size > 1:
                band_ds.name = f"{ds.name}_band{band:02d}"
            band_gdfs.append(ds2gdf(band_ds))

    if len(band_gdfs) > 1:
        gdf = merge_dfs(band_gdfs)
    elif len(band_gdfs) == 0:
        raise ValueError("No bands found in raster.")
    else:
        gdf = band_gdfs[0]

    return gdf


def merge_dfs(
    gdfs: list[Union[pd.DataFrame, gpd.GeoDataFrame]],
    how: str = "left",
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Merge GeoDataFrames on matching data

    Args:
        gdfs (list[Union[pd.DataFrame, gpd.GeoDataFrame]]): List of DataFrames to merge
            on a common column
        how (str, optional): Type of merge to be performed. Defaults to "inner".

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: Merged DataFrame
    """
    merged_gdf = reduce(lambda left, right: left.merge(right, how=how), gdfs)

    # For some reason the merge() method for dask geodataframes returns a pandas
    # dataframe when computed, so we need to compute it and then convert it back to a
    # dask geodataframe
    # merged_gdf = merged_gdf.compute()
    # merged_gdf = gpd.GeoDataFrame(merged_gdf, crs=crs, geometry=merged_gdf.geometry)
    merged_gdf = dgpd.from_dask_dataframe(merged_gdf, geometry=merged_gdf.geometry)

    return merged_gdf


def resample_raster(
    dataset: xr.DataArray, upscale_factor: float = 1 / 3
) -> xr.DataArray:
    """Resample a rioxarray dataset according to an upscale factor.

    Args:
        dataset (xarray.DataArray): Raster dataset
        upscale_factor (float, optional): Scaling factor. Upsamples if between 0 and 1.
        Defaults to 1/3.

    Returns:
        xarray.DataArray: Resampled dataset
    """
    height = int(dataset.rio.height * upscale_factor)
    width = int(dataset.rio.width * upscale_factor)
    resampled = dataset.rio.reproject(
        dataset.rio.crs, shape=(height, width), resampling=Resampling.bilinear
    )

    return resampled


def print_shapes(
    X: gpd.GeoDataFrame, Y: gpd.GeoDataFrame, rows_dropped: bool = True
) -> None:
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    if rows_dropped:
        rows = abs((Y.shape[0] - X.shape[0]))
        if rows == 0:
            print("Rows match\n")
        else:
            print("Rows dropped:", rows, "\n")


def drop_XY_NAs(
    XY: Union[gpd.GeoDataFrame, pd.DataFrame],
    X_cols: Union[pd.Index, str],
    Y_cols: Union[pd.Index, str],
    verbose: int = 0,
) -> Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.Index, Union[pd.Index, str]]:
    """Drop all rows and columns that contain all NAs in either X (predictors) or Y
    (response variables).

    Args:
        XY (Union[gpd.GeoDataFrame, pd.DataFrame]): Dataframe containing both X and Y
            variables
        X_cols (pd.Index): Index identifying the column(s) containing the predictors
        Y_cols (pd.Index, str): Index or string identifying the column(s) containing the response variable(s)
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.Index, pd.Index]: Original
            dataframe with full-NA rows/columns in the X and Y spaces dropped.
    """

    shape_before = XY.shape

    # Drop all rows that contain no response variable data at all
    XY = XY.dropna(axis=0, subset=Y_cols, how="all")
    # Drop all rows that contain no predictor data at all
    XY = XY.dropna(axis=0, subset=X_cols, how="all")
    # Drop all columns that contain no data at all
    XY = XY.dropna(axis=1, how="all")

    if (isinstance(Y_cols, str)) and (Y_cols not in XY.columns):
        raise ValueError(f"The single y column ({Y_cols}) contained all NAs!")

    if verbose:
        print(f"XY shape before dropping full-NA rows/cols: {shape_before}")
        shape_after = XY.shape
        rows_dropped = shape_before[0] - shape_after[0]
        message = ""
        message += f"XY shape after:                             {shape_after}"
        message += f"\n# of rows excluded: {rows_dropped} ({(rows_dropped / shape_before[0]) * 100:.2f}%)"
        dropped_X_cols = X_cols[~X_cols.isin(XY.columns)].values

        if (isinstance(Y_cols, str)) and (Y_cols in XY.columns):
            dropped_Y_cols = []
        else:
            dropped_Y_cols = Y_cols[~Y_cols.isin(XY.columns)].values

        if len(dropped_X_cols) == 0 and len(dropped_Y_cols) == 0:
            message += "\n\nNo columns were dropped."
        else:
            message += "\nEmpty columns:"
            message += f"\nX: {dropped_X_cols}"
            message += f"\nY: {dropped_Y_cols}"

        print(message)
    # Update X_cols and Y_cols accordingly to account for any dropped columns
    X_cols = X_cols[X_cols.isin(XY.columns)]
    Y_cols = Y_cols[Y_cols.isin(XY.columns)] if not isinstance(Y_cols, str) else Y_cols

    return XY, X_cols, Y_cols


def get_epsg(raster: Union[str, os.PathLike, xr.DataArray]) -> int:
    """Get the EPSG code of a raster.

    Args:
        raster (Union[str, os.PathLike, xr.DataArray]): Raster dataset

    Returns:
        int: EPSG code
    """
    dataset = validate_raster(raster)

    return int(dataset.rio.crs.to_epsg())


def validate_raster(raster: Union[str, os.PathLike, xr.DataArray]) -> xr.DataArray:
    """Validate a raster dataset.

    Args:
        raster (Union[str, os.PathLike, xr.DataArray]): Raster dataset

    Returns:
        xr.DataArray: Validated raster dataset
    """
    if isinstance(raster, (str, os.PathLike)):
        name = os.path.splitext(os.path.basename(raster))[0]
        dataset = xr.open_dataarray(raster, masked=True, default_name=name)
        if isinstance(dataset, list):
            dataset = dataset[0]
        return dataset

    if not isinstance(raster, xr.DataArray):
        raise TypeError("Raster is neither a filename nor a valid dataset.")

    return raster


def netcdf2gdf(da: xr.DataArray, data_label: str, name: str) -> gpd.GeoDataFrame:
    """Converts a netCDF dataset to a GeoDataFrame"""
    # Convert to a DataFrame
    df = da.to_dask_dataframe().reset_index()
    df = df.rename(columns={"lon": "x", "lat": "y"})
    df = df.dropna(subset=["x", "y"])
    if "months" in df.columns:
        df = df.drop(columns=["month"])
    df = df.rename(columns={data_label: name})
    geometry = dgpd.points_from_xy(df.x, df.y)
    df = dgpd.from_dask_dataframe(df, geometry=geometry)
    df = df.set_crs("EPSG:4326")
    return df


def ts_netcdf2gdfs(
    ds: Union[xr.Dataset, str, os.PathLike], ds_name: Optional[str] = None
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Converts a timeseries netCDF dataset to a GeoDataFrame with columns for each time step.

    Args:
        ds (Union[xr.Dataset, str, os.PathLike]): Dataset
        ds_name (Optional[str], optional): Name of the dataset. Defaults to None.

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: GeoDataFrame with columns for each time step

    """

    gdfs = []

    if isinstance(ds, (str, os.PathLike)):
        band_name = str(ds).split("_")[0].lower() + "_band"
        ds = xr.open_dataset(ds)

    data_label = str(list(ds.keys())[0])  # assume that the first variable is the data

    if data_label == "Band1":
        # In this case, the GDAL resampling resulted in the "month" dimension being
        # expanded into 12 separate data variables. This could be cleaned up as a TODO.
        for i in range(1, 13):
            data_label = f"Band{i}"
            da = ds[data_label]

            if ds_name:
                da_name = f"{ds_name}_{band_name}_m{i:02d}"
            else:
                da_name = f"{data_label}_m{i:02d}"
            df = netcdf2gdf(da, data_label, da_name)
            gdfs.append(df)
    else:
        for da in ds[data_label]:
            month = int(da.month.values)

            if ds_name:
                da_name = f"{ds_name}_{data_label}_m{month:02d}"
            else:
                da_name = f"{data_label}_m{month:02d}"

            df = netcdf2gdf(da, data_label, da_name)
            gdfs.append(df)

    if len(gdfs) > 1:
        gdf = merge_dfs(gdfs)
    else:
        gdf = gdfs[0]

    return gdf


def mask_oceans(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Masks oceans from a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with oceans masked.
    """
    land_mask = gpd.read_feather("./data/masks/land_mask_110m.feather")
    land_mask = land_mask.to_crs(gdf.crs)
    # if isinstance(gdf, dgpd.GeoDataFrame):
    #     gdf = gdf.compute()
    return gdf.clip(land_mask)
