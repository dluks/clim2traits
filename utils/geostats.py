from typing import Optional, Sequence

import geopandas as gpd
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import BallTree


def aoa(
    new_data: gpd.GeoDataFrame,
    training_data: gpd.GeoDataFrame,
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
    if len(training_data) <= 1:
        raise ValueError("Training data must contain more than one instance.")

    if len(training_data.columns) != len(new_data.columns):
        raise ValueError(
            "Number of columns in training data and new data must be the same."
        )

    if weights and (len(training_data.columns) != len(weights)):
        raise ValueError(
            "Number of columns in training data and weights must be the same."
        )

    # Scale data
    training_data = (training_data - np.mean(training_data)) / np.std(training_data)
    new_data = (new_data - np.mean(new_data)) / np.std(new_data)

    # Apply feature weights
    if weights:
        for col, weight in weights:
            training_data[col] *= weight
            new_data[col] *= weight

    # Calculate nearest training instance to test data, return Euclidean distances
    tree = BallTree(training_data)
    mindist, _ = tree.query(new_data, k=1, return_distance=True)

    # Build matrix of pairwise distances
    paired_distances = pdist(training_data)
    train_dist = squareform(paired_distances)
    np.fill_diagonal(train_dist, np.nan)

    # Remove data points that are within the same fold
    if fold_indices:
        # Get number of training instances in each fold
        instances_in_folds = [len(fold) for fold in fold_indices]
        instance_fold_id = np.repeat(
            np.arange(0, len(fold_indices)), instances_in_folds
        )

        # Create mapping between training instance and fold ID
        fold_indices = np.concatenate(fold_indices)
        folds = np.vstack((fold_indices, instance_fold_id)).T

        # Mask training points in same fold for DI measure calculation
        for i, row in enumerate(train_dist):
            mask = folds[:, 0] == folds[:, 0][i]
            train_dist[i, mask] = np.nan

    # Scale distance to nearest training point by average distance across training data
    train_dist_mean = np.nanmean(train_dist, axis=1)
    train_dist_avgmean = np.mean(train_dist_mean)
    mindist /= train_dist_avgmean

    # Define threshold for AOA
    train_dist_min = np.nanmin(train_dist, axis=1)
    # aoa_train_stats = np.quantile(
    #     train_dist_min / train_dist_avgmean,
    #     q=np.array([0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1]),
    # )
    thres = np.quantile(train_dist_min / train_dist_avgmean, q=thres)

    # We choose the AOA as the area where the DI does not exceed the threshold
    DIs = mindist.reshape(-1)
    masked_result = np.repeat(1, len(mindist))
    masked_result[DIs > thres] = 0

    return DIs, masked_result
