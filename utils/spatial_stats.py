from typing import Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacv
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import BallTree

from utils.dataset_tools import timer


def block_cv_splits(
    X: npt.NDArray,
    coords: pd.Series,
    grid_size: float,
    buffer_radius=0,
    n_groups: int = 10,
    random_state: int = 42,
    verbose: int = 0,
) -> Sequence:
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
):
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
    # geometry = new_df.pop("geometry")
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

    # Scale data
    print("Scaling data...")
    training_data = normalize(training_data)
    new_data = normalize(new_data)

    # Convert back to dataframes to apply feature weights by column
    training_data = pd.DataFrame(training_data, columns=data_cols)
    new_data = pd.DataFrame(new_data, columns=data_cols)

    # Apply feature weights
    if weights is not None:
        print("Applying feature weights...")
        training_data, new_data = map(
            lambda x: apply_weights(x, weights), [training_data, new_data]
        )

    print("Calculating nearest training instance...")
    mindist = nearest_dist(training_data, new_data)

    print("Calculating pairwise distances...")
    train_dist = train_dists(training_data)

    # Remove data points that are within the same fold
    print("Masking training points in same fold...")
    if fold_indices:
        folds = map_folds(fold_indices)
        train_dist = mask_folds(train_dist, folds)

    print("Calculating AOA...")
    DIs, masked_result = calc_aoa(mindist, train_dist, thres)

    new_df["DI"] = DIs
    new_df["AOA"] = masked_result

    return new_df


def normalize(data: npt.NDArray) -> npt.NDArray:
    """Normalize data to mean 0 and standard deviation 1."""
    return (data - np.mean(data)) / np.std(data)


def apply_weights(data: pd.DataFrame, weights: np.ndarray) -> npt.NDArray:
    """Apply feature weights to data."""
    for col, weight in weights:
        weight = float(weight)
        data[col] = data[col] * weight
    return data.to_numpy()


@timer
def nearest_dist(training_data, new_data):
    # Calculate nearest training instance to test data, return Euclidean distances
    tree = BallTree(training_data)
    mindist, _ = tree.query(new_data, k=1, return_distance=True)
    return mindist


@timer
def train_dists(training_data):
    # Build matrix of pairwise distances
    paired_distances = pdist(training_data)
    train_dist = squareform(paired_distances)
    np.fill_diagonal(train_dist, np.nan)
    return train_dist


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
                        print(f"Removing duplicate {duplicate} from fold {i}...")
                        fold = np.delete(fold, np.where(fold == duplicate), axis=0)
                        fold_indices[i] = fold
                        # mask = np.ones(len(fold), dtype=bool)
                        # mask[np.where(fold == duplicate)] = False
                        # fold_indices[i] = fold[mask]

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
