#!/usr/bin/env python3

import multiprocessing
from typing import Tuple

import geopandas as gpd
import numpy as np
import skgstat as skg
from scipy.spatial.distance import squareform
from sklearn.metrics.pairwise import haversine_distances

from TrainModelConfig import TrainModelConfig
from utils.data_retrieval import all_gdfs

# def haversine(p1, p2):
#     lat1, lon1 = p1
#     lat2, lon2 = p2

#     R = 6372.8  # Earth radius in kilometers

#     lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

#     dlon = lon2 - lon1
#     dlat = lat2 - lat1

#     a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2

#     c = 2 * np.arcsin(np.sqrt(a))

#     return R * c


def haversine(p1, p2):
    dists = haversine_distances(np.radians([p1, p2]))
    dists = dists * 6371000 / 1000
    dists = squareform(dists)
    return dists


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

    pool = multiprocessing.Pool(10)
    results = []

    for col in X.values.T:
        coords = all_coords[~np.isnan(col)]
        values = col[~np.isnan(col)]
        args = (coords, values)
        results.append(pool.apply_async(calculate_range, (args,)))

    for result in results:
        ranges.append(result.get())
        
    pool.close()
    pool.join()

    ranges = np.asarray(ranges)
    med = np.median(ranges)

    return ranges, med


if __name__ == "__main__":
    config = TrainModelConfig()

    X_fns = config.WC_fns + config.MODIS_fns + config.soil_fns
    Y_fns = config.iNat_fns

    X = all_gdfs(X_fns)
    Y = all_gdfs(Y_fns)

    bio_13 = X.loc[:, ["bio_13" in x for x in X.columns]].values
    bio_14 = X.loc[:, ["bio_14" in x for x in X.columns]].values
    X["wc2.1_10m_bio_13-14"] = bio_13 - bio_14

    X = X.drop(columns=["x", "y", "band", "spatial_ref"])
    Y = Y.drop(columns=["x", "y", "band", "spatial_ref"])

    Y = Y[["geometry", "iNat_SLA_05deg_ln"]]

    # Drop response variable NAs
    Y = Y.dropna()

    Y.head(5)

    X = X.loc[X["geometry"].isin(Y["geometry"])]

    X = X.dropna(subset=X.columns.difference(["geometry"]), how="all")

    Y = Y.loc[Y["geometry"].isin(X["geometry"])]
    
    X = X.dropna(axis=1, how="all")

    XYs = X["geometry"]
    ranges, med_range = vgm_ranges(XYs, X[X.columns.difference(["geometry"])])

    np.save("../vgm_ranges.npy", ranges)
