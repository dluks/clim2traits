################################
# Geodata utils
################################
import os
from functools import reduce
from typing import Tuple, Union

import geopandas as gpd
import pandas as pd
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling


def tif2gdf(raster: Union[str, xr.DataArray]) -> gpd.GeoDataFrame:
    """Converts a GeoTIFF to a GeoPandas data frame. Accepts either the filename of a
    GeoTiff or an xr dataset.

    Args:
        raster (str || xarray.DataArray): Filename of the GeoTIFF raster or the opened
        raster itself.

    Returns:
        geopandas.GeoDataFrame: GeoPandas data frame
    """
    dataset = validate_raster(raster)

    if len(dataset) > 1:
        raise ValueError("raster must be a single-band dataset.")

    df = dataset.squeeze().to_dataframe().reset_index()
    geometry = gpd.points_from_xy(df.x, df.y)
    gdf = gpd.GeoDataFrame(df, crs=dataset.rio.crs, geometry=geometry)

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
        geo_col (str, optional): Name of the geometry column. Defaults to "geometry".

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: Merged DataFrame
    """
    merged_gdf = reduce(lambda left, right: pd.merge(left, right, how=how), gdfs)

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
        rows = abs(Y.shape[0] - X.shape[0])
        if rows == 0:
            print("Rows match\n")
        else:
            print("Rows dropped:", rows, "\n")


def drop_XY_NAs(
    XY: Union[gpd.GeoDataFrame, pd.DataFrame, gpd.GeoSeries, pd.Series],
    X_cols: pd.Index,
    Y_cols: Union[pd.Index, str],
    verbose: int = 0,
) -> Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.Index, pd.Index]:
    """Drop all rows and columns that contain all NAs in either X (predictors) or Y
    (response variables).

    Args:
        XY (Union[gpd.GeoDataFrame, pd.DataFrame]): Dataframe containing both X and Y
            variables
        X_cols (pd.Index): Index identifying the column(s) containing the predictors
        Y_cols (pd.Index, str): Index or string identifying the column(s) containing
            the response variable(s)
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.Index, pd.Index]: Original
            dataframe with full-NA rows/columns in the X and Y spaces dropped.
    """

    if verbose:
        shape_before = XY.shape
        print(f"XY shape before dropping full-NA rows/cols: {shape_before}")
    # Drop all rows that contain no response variable data at all
    XY = XY.dropna(axis=0, subset=Y_cols, how="all")
    # Drop all rows that contain no predictor data at all
    XY = XY.dropna(axis=0, subset=X_cols, how="all")
    # Drop all columns that contain no data at all
    XY = XY.dropna(axis=1, how="all")

    if (isinstance(Y_cols, str)) and (Y_cols not in XY.columns):
        raise ValueError(f"The single y column ({Y_cols}) contained all NAs!")

    if verbose:
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

def validate_raster(
    raster: Union[str, os.PathLike, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray, list[xr.Dataset]]:
    """Validate a raster dataset.

    Args:
        raster (Union[str, os.PathLike, xr.DataArray]): Raster dataset

    Returns:
        xr.DataArray: Validated raster dataset
    """
    if isinstance(raster, (str, os.PathLike)):
        name = os.path.splitext(os.path.basename(raster))[0]
        dataset = riox.open_rasterio(raster, masked=True, default_name=name)
        return dataset

    if not isinstance(raster, xr.DataArray):
        raise TypeError("Raster is neither a filename nor a valid dataset.")

    return raster
