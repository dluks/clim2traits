################################
# Geodata utils
################################
import os
from functools import reduce
from typing import Union

import geopandas as gpd
import pandas as pd
import rioxarray as rx
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
    if isinstance(raster, str):
        name = os.path.splitext(os.path.basename(raster))[0]
        raster = rx.open_rasterio(raster, masked=True)
        raster.name = name
    elif not isinstance(raster, xr.DataArray):
        raise TypeError("raster is neither a filename or an xr dataset.")

    df = raster.squeeze().to_dataframe().reset_index()
    geometry = gpd.points_from_xy(df.x, df.y)
    gdf = gpd.GeoDataFrame(df, crs=raster.rio.crs, geometry=geometry)

    return gdf


def merge_gdfs(
    gdfs: list[gpd.GeoDataFrame], how: str = "inner", geo_col: str = "geometry"
) -> gpd.GeoDataFrame:
    """Merge GeoDataFrames on matching data

    Args:
        gdfs (list): GeoDataFrames to be merged
        how (str, optional): Type of merge to be performed. Defaults to "inner".
        geo_col (str, optional): Name of the geometry column. Defaults to "geometry".

    Returns:
        geopandas.GeoDataFrame: Merged GeoDataFrame
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
