from typing import Any, Generator, Optional, Sequence, Tuple, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacv
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise_distances, pairwise_distances_chunked
from verstack import NaNImputer

from utils.dataset_tools import timer


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
            f"Imputed data shape ({imputed.shape}) does not match original data shape ({data.shape})"
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
    # Calculate nearest training instance to test data, return Euclidean distances
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
    # Calculate nearest training instance to test data, return Euclidean distances
    distances = pairwise_distances(
        new_data, training_data, metric="euclidean", n_jobs=20
    ).min(axis=1)

    return distances


@timer
def calc_train_dists(training_data):
    # Build matrix of pairwise distances
    train_dists = pairwise_distances(training_data, metric="euclidean", n_jobs=20)
    np.fill_diagonal(train_dists, np.nan)
    return train_dists


def map_folds(fold_indices):
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
