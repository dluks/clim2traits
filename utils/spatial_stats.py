from multiprocessing import Pool
from pathlib import Path
from typing import Any, Generator, Optional, Sequence, Tuple, Union

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import rioxarray as riox
import spacv
import statsmodels.api as statmod
import xarray as xr
from pyproj import Proj
from shapely.geometry import shape
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from tqdm import tqdm
from verstack import NaNImputer

from utils.dataset_tools import timer
from utils.geodata import (
    back_transform_trait,
    get_trait_id_from_data_name,
    num_to_str,
    open_raster,
    xr_to_gdf,
)


def block_cv_splits(
    X: npt.NDArray,
    coords: pd.Series,
    grid_size: float,
    buffer_radius: Union[float, int] = 0,
    n_groups: int = 10,
    random_state: int = 42,
    verbose: int = 0,
) -> Generator[tuple[npt.NDArray, Any], Any, None]:
    """Define spatial folds for cross-validation

    Args:
        X (NDArray): X training data
        coords (pd.Series): Coordinates of training data
        grid_size (float): Size of grid in degrees
        buffer_radius (float, optional): Buffer radius in degrees. Defaults to 0.01.
        n_groups (int, optional): Number of groups to split into. Defaults to 10.
        random_state (int, optional): Random state. Defaults to 42.
        verbose (int, optional): Verbosity. Defaults to 0.

    Returns:
        Sequence: Spatial folds iterator
    """
    if verbose == 1:
        print("Defining spatial folds...")
    tiles_x = int(np.round(360 / grid_size))
    tiles_y = int(np.round(180 / grid_size))

    hblock = spacv.HBLOCK(
        tiles_x,
        tiles_y,
        shape="hex",
        method="optimized_random",
        buffer_radius=buffer_radius,
        n_groups=n_groups,
        data=X,
        n_sims=50,
        distance_metric="haversine",
        random_state=random_state,
    )

    splits = hblock.split(coords)

    return splits


def aoa(
    new_df: Union[gpd.GeoDataFrame, pd.DataFrame],
    training_df: Union[gpd.GeoDataFrame, pd.DataFrame],
    weights: Optional[np.ndarray] = None,
    thres: float = 0.95,
    fold_indices: Optional[Sequence] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Area of Applicability (AOA) measure for spatial prediction models from
    Meyer and Pebesma (2020). The AOA defines the area for which, on average,
    the cross-validation error of the model applies, which is crucial for
    cases where spatial predictions are used to inform decision-making.

    Adapted from spacv by Sam Comber (https://github.com/SamComber/spacv).

    Parameters
    ----------
    new_data : GeoDataFrame
        A GeoDataFrame containing unseen data to measure AOA for.
    training_data : GeoDataFrame
        A GeoDataFrame containing the features used for model training.
    weights : array, default=None
        Array of weights corresponding to the feature importance of each predictor. Each
        item in the array should a 1x2 array containing the feature name and its weight.
    thres : default=0.95
        Threshold used to identify predictive area of applicability.
    fold_indices : iterable, default=None
        iterable consisting of training indices that identify instances in the
        folds.
    Returns
    -------
    DIs : array
        Array of disimimilarity scores between training_data for new_data points.
    masked_result : array
        Binary mask that occludes points outside predictive area of applicability.
    """
    # Only keep columns that are in both training and new data
    common_cols = training_df.columns.intersection(new_df.columns)
    training_df = training_df[common_cols]
    new_df = new_df[common_cols]

    # Find columns that contain all NaNs in either training or new and drop them from
    # both dfs as well as weights
    nan_cols = training_df.columns[
        training_df.isna().all(axis=0) | new_df.isna().all(axis=0)
    ]
    print(f"Dropping {nan_cols}...")
    training_df = training_df.drop(nan_cols, axis=1)
    new_df = new_df.drop(nan_cols, axis=1)
    if weights is not None:
        weights = weights[~np.isin(weights[:, 0], nan_cols)]

    data_cols = training_df.columns
    training_data = training_df[data_cols].to_numpy()
    new_data = new_df[data_cols].to_numpy()

    if len(training_data) <= 1:
        raise ValueError("Training data must contain more than one instance.")

    if len(training_df.columns.difference(["geometry"])) != len(
        new_df.columns.difference(["geometry"])
    ):
        raise ValueError(
            f"Number of columns in training dataframe ({len(training_df.columns)}) \
and new dataframe ({len(new_df.columns)}) must be the same."
        )

    if weights is not None and (len(data_cols) != len(weights)):
        raise ValueError(
            "Number of columns in training data and weights must be the same."
        )
    # Scale data =============================================================
    print("Scaling data...")
    training_data = normalize(training_data)
    new_data = normalize(new_data)

    # Impute missing values ==================================================
    new_data = pd.DataFrame(new_data, columns=data_cols)

    # Check if new data dataframe contains any NaNs
    if new_data.isnull().values.any():
        print("Imputing missing values in new data...")
        new_data = impute_missing(new_data)
    else:
        print("No missing values in new data. Not imputing.")

    training_data = pd.DataFrame(training_data, columns=data_cols)

    if training_data.isnull().values.any():
        print("Imputing missing values in training data...")
        training_data = impute_missing(training_data)
    else:
        print("No missing values in training data. Not imputing.")

    # Apply feature weights ==================================================
    if weights is not None:
        print("Applying feature weights...")
        training_data, new_data = map(
            lambda x: apply_weights(x, weights), [training_data, new_data]
        )

    training_data = training_data.to_numpy()
    new_data = new_data.to_numpy()

    # Calculate nearest training instance ====================================
    print("Calculating nearest training instance...")
    mindist = nearest_dist_chunked(training_data, new_data)

    # Calculate pairwise distances ===========================================
    print("Calculating pairwise distances in training data...")
    train_dists = calc_train_dists(training_data)

    # Mask folds =============================================================
    print("Masking training points in same fold...")
    if fold_indices:
        folds = map_folds(fold_indices)
        train_dists = mask_folds(train_dists, folds)

    print("Calculating AOA...")
    DIs, masked_result = calc_aoa(mindist, train_dists, thres)

    return DIs, masked_result


def normalize(data: npt.NDArray) -> npt.NDArray:
    """Normalize data to mean 0 and standard deviation 1."""
    return (data - np.nanmean(data)) / np.nanstd(data)


def apply_weights(data: pd.DataFrame, weights: np.ndarray) -> pd.DataFrame:
    """Apply feature weights to data."""
    for col, weight in weights:
        weight = float(weight)
        data[col] = data[col] * weight
    return data


def impute_missing(data: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Impute missing values."""

    first_imputer = NaNImputer(verbose=verbose)
    imputed = first_imputer.impute(data)

    # Re-add dropped columns from first imputation
    dropped_cols = data.columns.difference(imputed.columns)
    imputed = pd.concat([imputed, data[dropped_cols]], axis=1)

    # Return order of columns to match original data
    imputed = imputed[data.columns]

    # Confirm that the shape of the imputed data is the same as the original
    if imputed.shape != data.shape:
        raise ValueError(
            f"Imputed data shape ({imputed.shape}) does not match original data shape"
            f"({data.shape})"
        )

    # Get columns that are completely empty and drop them
    empty_cols = imputed.columns[imputed.isna().all(axis=0)]
    imputed = imputed.drop(columns=empty_cols)

    # Fill remaining missing values with simple imputer
    second_imputer = SimpleImputer(strategy="mean")
    imputed = second_imputer.fit_transform(imputed)

    # convert back to dataframe
    imputed = pd.DataFrame(imputed, columns=data.columns.difference(empty_cols))
    # add back empty columns with NaNs
    imputed = pd.concat([imputed, data[empty_cols].reset_index(drop=True)], axis=1)

    # Confirm that the shape of the imputed data is the same as the original
    if imputed.shape != data.shape:
        raise ValueError(
            f"Imputed data shape ({imputed.shape}) does not match original data shape ({data.shape})"
        )

    # Confirm that there are no NaNs in the imputed data
    if imputed.isnull().values.any():
        raise ValueError("Imputed data contains NaNs.")

    return imputed


@timer
def nearest_dist_chunked(training_data, new_data):
    """Calculate nearest training instance to test data, return Euclidean distances"""
    distances = np.concatenate(
        [
            chunk.min(axis=1)
            for chunk in pairwise_distances_chunked(
                new_data, training_data, metric="euclidean", n_jobs=20
            )
        ]
    )

    return distances


@timer
def nearest_dist(training_data, new_data):
    """Calculate nearest training instance to test data, return Euclidean distances"""
    distances = pairwise_distances(
        new_data, training_data, metric="euclidean", n_jobs=20
    ).min(axis=1)

    return distances


@timer
def calc_train_dists(training_data):
    """Build matrix of pairwise distances"""
    train_dists = pairwise_distances(training_data, metric="euclidean", n_jobs=20)
    np.fill_diagonal(train_dists, np.nan)
    return train_dists


def map_folds(fold_indices):
    """Map training instances to fold IDs"""
    # First remove any within-fold duplicates
    fold_indices = [np.unique(fold) for fold in fold_indices]

    # Check for duplicate ids across folds
    unique, counts = np.unique(np.concatenate(fold_indices), return_counts=True)
    duplicates = unique[counts > 1]

    # If duplicates exist, remove all but the first instance of each duplicate
    if len(duplicates) > 0:
        for duplicate in duplicates:
            count = 0
            for i, fold in enumerate(fold_indices):
                if duplicate in fold:
                    count += 1
                    if count >= 2:
                        fold = np.delete(fold, np.where(fold == duplicate), axis=0)
                        fold_indices[i] = fold

    # Get number of training instances in each fold
    instances_in_folds = [len(fold) for fold in fold_indices]
    instance_fold_id = np.repeat(np.arange(0, len(fold_indices)), instances_in_folds)

    # Create mapping between training instance and fold ID
    fold_indices = np.concatenate(fold_indices)

    folds = np.vstack((fold_indices, instance_fold_id)).T

    return folds


def mask_folds(train_dist, folds):
    """Mask training points in same fold for DI measure calculation"""
    # sort folds by point id for easier masking
    folds = folds[np.argsort(folds[:, 0])]

    # Mask training points in same fold for DI measure calculation
    for i, _ in enumerate(train_dist):
        point_fold = folds[i, 1]
        mask = folds[:, 1] == point_fold
        train_dist[i, mask] = np.nan
    return train_dist


@timer
def calc_aoa(mindist, train_dist, thres):
    """Calculate AOA"""
    # Scale distance to nearest training point by average distance across training data
    train_dist_mean = np.nanmean(train_dist, axis=1)
    train_dist_avgmean = np.mean(train_dist_mean)
    mindist /= train_dist_avgmean

    # Define threshold for AOA
    train_dist_min = np.nanmin(train_dist, axis=1)
    thres = np.quantile(train_dist_min / train_dist_avgmean, q=thres)

    # We choose the AOA as the area where the DI does not exceed the threshold
    DIs = mindist.reshape(-1)
    masked_result = np.repeat(1, len(mindist))
    masked_result[DIs > thres] = 0

    return DIs, masked_result


def lat_weights(lat_unique, deg):
    """Calculate weights for each latitude band based on area of grid cells.
    Source: https://sojwolf.github.io/iNaturalist_traits/Chapter_14_Closer_Look_at_Schiller_map.html
    """

    # determine weights per grid cell based on longitude
    # keep only one exemplary cell at each distance from equator
    # weights per approximated area of grid size depending on distance from equator

    # make dictionary
    weights = {}

    for j in lat_unique:
        # the four corner points of the grid cell

        p1 = (0, j + (deg / 2))
        p2 = (deg, j + (deg / 2))
        p3 = (deg, j - (deg / 2))
        p4 = (0, j - (deg / 2))

        # Calculate polygon surface area
        # https://stackoverflow.com/questions/4681737/how-to-calculate-the-area-of-a-polygon-on-the-earths-surface-using-python

        # Define corner points
        co = {"type": "Polygon", "coordinates": [[p1, p2, p3, p4]]}
        lat_1 = p1[1]
        lat_2 = p3[1]
        lat_0 = (p1[1] + p3[1]) / 2
        lon_0 = deg / 2

        # Caveat: Connot go accross equator
        value1 = abs(lat_1 + lat_2)
        value2 = abs((lat_1) + abs(lat_2))

        # if grid cell overlaps equator:
        if value1 < value2:
            lat_1 = p1[1]
            lat_2 = 0
            lat_0 = (p1[1] + lat_2) / 2
            lon_0 = deg / 2

            # Projection equal area used: https://proj.org/operations/projections/aea.html
            projection_string = (
                "+proj=aea +lat_1="
                + str(lat_1)
                + " +lat_2="
                + str(lat_2)
                + " +lat_0="
                + str(lat_0)
                + " +lon_0="
                + str(lon_0)
            )
            lon, lat = zip(*co["coordinates"][0])

            pa = Proj(projection_string)

            # Only coercing to tuple bc pylint doesn't seem to respect the Proj object's
            # __call__ type hints
            x, y = tuple(pa(lon, lat))
            cop = {"type": "Polygon", "coordinates": [zip(x, y)]}

            area = (shape(cop).area / 1000000) * 2

        # if grid cell is on one side of equator:
        else:
            # Projection equal area used: https://proj.org/operations/projections/aea.html
            projection_string = (
                "+proj=aea +lat_1="
                + str(lat_1)
                + " +lat_2="
                + str(lat_2)
                + " +lat_0="
                + str(lat_0)
                + " +lon_0="
                + str(lon_0)
            )
            lon, lat = zip(*co["coordinates"][0])

            pa = Proj(projection_string)
            # Only coercing to tuple bc pylint doesn't seem to respect the Proj object's
            # __call__ type hints
            x, y = tuple(pa(lon, lat))
            cop = {"type": "Polygon", "coordinates": [zip(x, y)]}

            area = shape(cop).area / 1000000

        # set coord to center of grid cell
        coord = j

        # map area to weights dictionary
        weights[coord] = area

    # convert area into proportion with area/max.area:
    max_area = max(weights.values())

    for key in weights:
        weights[key] = weights[key] / max_area

    return weights


def weighted_r(
    df: gpd.GeoDataFrame,
    col_1: str,
    col_2: str,
    col_lat: str,
    weights: dict,
    r2: bool = False,
):
    """Calculate weighted correlation between two columns of a dataframe.
    Source: https://sojwolf.github.io/iNaturalist_traits/Chapter_14_Closer_Look_at_Schiller_map.html
    """
    # map weights to dataframe
    df["Weights"] = df[col_lat].map(weights)

    # drop nan
    df = df.dropna()

    # calculate weighted correlation
    # https://www.statsmodels.org/stable/generated/statsmodels.stats.weightstats.DescrStatsW.html
    d1 = statmod.stats.DescrStatsW(df[[col_1, col_2]], df["Weights"])

    corr = d1.corrcoef[0][1]

    # optional
    # calculate r2
    if r2:
        corr = corr**2

    return corr


def compare_gdfs(
    gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame, weights: Optional[dict] = None
) -> float:
    """Calculate the correlation coefficient between two GeoDataFrames. Assumes that the
    two GeoDataFrames consist of a geometry column and a single data column.
    """
    col1 = gdf1.columns.difference(["geometry"]).values[0]
    col2 = gdf2.columns.difference(["geometry"]).values[0]
    merged = gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")
    merged = merged.dropna()
    merged["x"] = merged.geometry.x
    merged["y"] = merged.geometry.y

    if weights is not None:
        corr = weighted_r(merged, col1, col2, "y", weights)
    else:
        corr = merged[col1].corr(merged[col2])
    return corr


def trait_correlation(
    trait_pred_dir: Path, grid_res: Union[int, float], pft: str
) -> Tuple[str, float]:
    """Get the correlation between trait predictions and sPlot maps for a given trait."""
    trait = trait_pred_dir.parent.name
    trait_id = get_trait_id_from_data_name(trait)
    trait_prediction_fn = list(trait_pred_dir.glob("*.parq"))[0]

    if grid_res == 0.01:
        cols = dgpd.read_parquet(trait_prediction_fn).columns.values
        trait_prediction = dgpd.read_parquet(
            trait_prediction_fn, columns=cols[:2]
        ).compute()

        # Round coordinates to 3 decimal places to account for floating point errors
        x = trait_prediction.geometry.x.round(3)
        y = trait_prediction.geometry.y.round(3)
        trait_prediction.geometry = gpd.points_from_xy(x, y)
        del x, y
    else:
        cols = gpd.read_parquet(trait_prediction_fn).columns.values
        trait_prediction = gpd.read_parquet(trait_prediction_fn, columns=cols[:2])

    if trait.endswith("_ln"):
        trait_prediction = back_transform_trait(trait_prediction)

    splot_dir = Path("GBIF_trait_maps/global_maps", pft, f"{num_to_str(grid_res)}deg")

    if grid_res >= 0.5:
        splot_fn = list(splot_dir.glob(f"sPlot*_X{trait_id}_*.grd"))[0]
        splot_ds = open_raster(splot_fn, masked=True).sel(band=2)
    else:
        splot_dir = splot_dir / "05_range"
        splot_fn = list(splot_dir.glob(f"sPlot*_X{trait_id}_*.tif"))[0]
        splot_ds = open_raster(splot_fn, masked=True).squeeze()

    if isinstance(splot_ds, list):
        raise ValueError("Multiple sPlot files found.")

    splot_ds.name = f"X{trait_id}"
    splot_df = xr_to_gdf(splot_ds)

    # Round coordinates to 3 decimal places to account for floating point errors
    x = splot_df.geometry.x.round(3)
    y = splot_df.geometry.y.round(3)
    splot_df.geometry = gpd.points_from_xy(x, y)
    del x, y

    # get latitude weights
    weights = lat_weights(splot_df.geometry.y.unique(), grid_res)
    corr = compare_gdfs(trait_prediction, splot_df, weights=weights)
    return f"X{trait_id}", corr


def splot_correlations(
    grid_res: Union[int, float], model_res: Union[int, float], pft: str
) -> Tuple[dict, dict]:
    """Get the correlations between trait predictions and sPlot maps for all traits in
    a given PFT and of a given resolution in degrees."""
    predictor_dataset = "MOD09GA.061_ISRIC_soil_WC_BIO_VODCA"
    nan_strat = "nan-strat=any_thr=0.5"
    dataset_name = f"{predictor_dataset}_{grid_res:g}_deg_{nan_strat}"

    dataset_dir_name = f"{'tiled_5x5_deg_' if grid_res == 0.01 else ''}{dataset_name}"

    prediction_dir = Path(
        "results/predictions",
        f"{num_to_str(model_res)}deg_models",
        dataset_dir_name,
        pft,
    )

    gbif_dirs = list(prediction_dir.glob("TRYgapfilled*/GBIF"))
    splot_dirs = list(prediction_dir.glob("TRYgapfilled*/sPlot"))

    if grid_res == 0.01:
        corr_table_gbif = {}
        corr_table_splot = {}

        for trait_dir in tqdm(gbif_dirs):
            trait, corr = trait_correlation(trait_dir, grid_res, pft)
            corr_table_gbif[trait] = corr

        for trait_dir in tqdm(splot_dirs):
            trait, corr = trait_correlation(trait_dir, grid_res, pft)
            corr_table_splot[trait] = corr
    else:
        with Pool() as pool:
            results_gbif = pool.starmap(
                trait_correlation,
                [
                    (trait_dir, grid_res, pft)
                    for trait_dir in prediction_dir.glob("TRYgapfilled*/GBIF")
                ],
            )
            results_splot = pool.starmap(
                trait_correlation,
                [
                    (trait_dir, grid_res, pft)
                    for trait_dir in prediction_dir.glob("TRYgapfilled*/sPlot")
                ],
            )

        corr_table_gbif = dict(results_gbif)
        corr_table_splot = dict(results_splot)

    return corr_table_gbif, corr_table_splot


def splot_correlation_old(
    gdf: gpd.GeoDataFrame, trait_id: str, trait_name: str, splot_set: str = "orig"
) -> float:
    """Get the correlation between trait predictions and sPlot maps for a given trait.

    Args:
        gdf (gpd.GeoDataFrame): Trait predictions
        trait_id (str): Trait ID
        trait_name (str): Trait name
        splot_set (str, optional): sPlot dataset (one of ["orig", "05_range"], case-insensitive).

    Returns:
        float: Pearson correlation coefficient
    """
    splot = riox.open_rasterio(
        f"data/splot/0.01_deg/{splot_set.lower()}/sPlot_{trait_id}_0.01deg.tif",
        masked=True,
    )
    splot.name = trait_name
    splot = xr_to_gdf(splot)
    splot_corr = compare_gdfs(gdf, splot)

    return splot_corr


def compare_gdf_to_grid(
    gdf: gpd.GeoDataFrame,
    grid: xr.DataArray,
    gdf_name: str,
    grid_name: str,
) -> np.float64:
    """Calculate the correlation coefficient between a GeoDataFrame and a raster grid."""
    # Ensure that the grids share the same CRS
    grid = grid.rio.reproject("EPSG:4326")

    # Convert to GDFs
    grid.name = grid_name
    grid = xr_to_gdf(grid).dropna(subset=[grid_name])

    # Merge the two geodataframes on the geometry column such that only matching
    # geometries are retained
    merged = gpd.sjoin(gdf, grid, how="inner", predicate="intersects")

    corr = merged[gdf_name].corr(merged[grid_name])
    return corr


def compare_grids(
    grid1: xr.DataArray, grid2: xr.DataArray, grid1_name: str, grid2_name: str
) -> np.float64:
    """Calculate the correlation coefficient between two raster grids."""
    # Ensure that the grids share the same CRS
    grid1 = grid1.rio.reproject("EPSG:4326")
    grid2 = grid2.rio.reproject("EPSG:4326")

    # Convert to GDFs
    grid1.name = grid1_name
    grid2.name = grid2_name
    grid1 = xr_to_gdf(grid1).dropna(subset=[grid1_name])
    grid2 = xr_to_gdf(grid2).dropna(subset=[grid2_name])

    # Merge the two geodataframes on the geometry column such that only matching
    # geometries are retained
    merged = gpd.sjoin(grid1, grid2, how="inner", op="intersects")
    corr = merged[grid1_name].corr(merged[grid2_name])
    return corr
