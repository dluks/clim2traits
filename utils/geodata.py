################################
# Geodata utils
################################
import multiprocessing
import os
from functools import reduce

# from math import asin, cos, radians, sin, sqrt
from typing import Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as rx
import skgstat as skg
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


# def vgm_ranges(XYs: gpd.GeoSeries, X: gpd.GeoDataFrame) -> Tuple[np.ndarray, float]:
#     """Calculates an spherical experimental variogram for each predictor in the data
#     frame and returns their respective ranges as well as the median range.

#     Args:
#         XYs (gpd.GeoSeries): XY positions in WKT format
#         X (gpd.GeoDataFrame): Data frame containing corresponding values

#     Returns:
#         Tuple[np.ndarray, float]: _description_
#     """
#     all_coords = np.asarray(list(map(lambda x: (x.x, x.y), XYs)))

#     ranges = []

#     for col in tqdm(X.values.T):
#         coords = all_coords[~np.isnan(col)]
#         values = col[~np.isnan(col)]
#         V = skg.Variogram(coordinates=coords, values=values)
#         ranges.append(V.parameters[0])

#     ranges = np.asarray(ranges)
#     med = np.median(ranges)

#     return ranges, med


def haversine(p1, p2):
    # # my adaption to fit the function signature
    # lat1, lon1 = p1
    # lat2, lon2 = p2

    # # original
    # R = 6372.8  # Earth radius in kilometers

    # dLat = radians(lat2 - lat1)
    # dLon = radians(lon2 - lon1)
    # lat1 = radians(lat1)
    # lat2 = radians(lat2)

    # a = sin(dLat / 2)**2 + cos(lat1) * cos(lat2) * sin(dLon / 2)**2
    # c = 2 * asin(sqrt(a))

    # return R * c

    lat1, lon1 = p1
    lat2, lon2 = p2

    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6367 * c
    return km


def calculate_range(args):
    coords, values = args
    V = skg.Variogram(coordinates=coords, values=values, dist_func=haversine)
    return V.parameters[0]


def vgm_ranges(XYs: gpd.GeoSeries, X: gpd.GeoDataFrame) -> Tuple[np.ndarray, float]:
    """Calculates a spherical experimental variogram for each predictor in the data
    frame and returns their respective ranges as well as the median range.

    Args:
        XYs (gpd.GeoSeries): XY positions in WKT format
        X (gpd.GeoDataFrame): Data frame containing corresponding values

    Returns:
        Tuple[np.ndarray, float]: Ranges of each variogram and the median range
    """
    all_coords = np.asarray(list(map(lambda x: (x.x, x.y), XYs)))

    ranges = []

    pool = multiprocessing.Pool()
    results = []

    for col in X.values.T:
        coords = all_coords[~np.isnan(col)]
        values = col[~np.isnan(col)]
        args = (coords, values)
        results.append(pool.apply_async(calculate_range, (args,)))

    pool.close()
    pool.join()

    for result in results:
        ranges.append(result.get())

    ranges = np.asarray(ranges)
    med = np.median(ranges)

    return ranges, med
