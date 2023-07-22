import time
import types
from typing import Optional, Sequence, Union

import geopandas as gpd
import numpy as np
import numpy.typing as npt
import pandas as pd
import spacv
from scipy.spatial.distance import pdist, squareform
from sklearn.experimental import enable_iterative_imputer  # type: ignore
from sklearn.impute import IterativeImputer
from sklearn.neighbors import BallTree


def block_cv_splits(
    X: npt.NDArray,
    coords: pd.Series,
    grid_size: float,
    buffer_radius=0.01,
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
    new_data: Union[gpd.GeoDataFrame, pd.DataFrame],
    training_data: Union[gpd.GeoDataFrame, pd.DataFrame],
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
    # new_data_geometry = new_data["geometry"].copy()
    training_data = training_data[training_data.columns.difference(["geometry"])].copy()
    new_data = new_data[new_data.columns.difference(["geometry"])].copy()

    if len(training_data) <= 1:
        raise ValueError("Training data must contain more than one instance.")

    if len(training_data.columns) != len(new_data.columns):
        raise ValueError(
            "Number of columns in training data and new data must be the same."
        )

    if weights is not None and (len(training_data.columns) != len(weights)):
        raise ValueError(
            "Number of columns in training data and weights must be the same."
        )

    print("Scaling data...")
    start = time.time()
    # Scale data
    training_data = (training_data - np.mean(training_data)) / np.std(training_data)
    new_data = (new_data - np.mean(new_data)) / np.std(new_data)
    end = time.time()
    print(f"Scaling took {end - start} seconds")

    print("Imputing missing values...")
    start = time.time()
    # Impute the mising values
    imp = IterativeImputer(max_iter=10, random_state=42)
    end = time.time()
    print(f"Imputing took {end - start} seconds")

    print("Fitting imputer...")
    start = time.time()
    # Fit imputer on the larger dataset
    if len(training_data) > len(new_data):
        imp.fit(training_data)
    else:
        imp.fit(new_data)
    end = time.time()
    print(f"Fitting took {end - start} seconds")

    print("Transforming data...")
    start = time.time()
    # Transform both datasets
    training_data_imp = imp.transform(training_data)
    new_data_imp = imp.transform(new_data)
    end = time.time()
    print(f"Transforming took {end - start} seconds")

    # Convert back to dataframes
    training_data = pd.DataFrame(training_data_imp, columns=training_data.columns)
    new_data = pd.DataFrame(new_data_imp, columns=new_data.columns)

    # Apply feature weights
    if weights is not None:
        print("Applying feature weights...")
        start = time.time()
        for col, weight in weights:
            weight = float(weight)
            training_data[col] = training_data[col] * weight
            new_data[col] = new_data[col] * weight
        end = time.time()
        print(f"Applying weights took {end - start} seconds")

    print("Dropping nans...")
    start = time.time()
    # Remove rows with nans for BallTree
    training_data = training_data.dropna()
    new_data = new_data.dropna()
    end = time.time()
    print(f"Dropping nans took {end - start} seconds")

    print("Calculating nearest training instance...")
    start = time.time()
    # Calculate nearest training instance to test data, return Euclidean distances
    tree = BallTree(training_data)
    mindist, _ = tree.query(new_data, k=1, return_distance=True)
    end = time.time()
    print(f"Calculating nearest took {end - start} seconds")

    print("Calculating pairwise distances...")
    start = time.time()
    # Build matrix of pairwise distances
    paired_distances = pdist(training_data)
    train_dist = squareform(paired_distances)
    np.fill_diagonal(train_dist, np.nan)
    end = time.time()
    print(f"Calculating pairwise took {end - start} seconds")

    # Remove data points that are within the same fold
    if fold_indices:
        if isinstance(fold_indices, types.GeneratorType):
            fold_indices = list(fold_indices)

        print("Masking training points in same fold...")
        start = time.time()
        # Get number of training instances in each fold
        instances_in_folds = [len(fold) for fold in fold_indices]
        instance_fold_id = np.repeat(
            np.arange(0, len(fold_indices)), instances_in_folds
        )

        # Create mapping between training instance and fold ID
        fold_indices = np.concatenate(fold_indices)
        folds = np.vstack((fold_indices, instance_fold_id)).T

        # Mask training points in same fold for DI measure calculation
        for i, _ in enumerate(train_dist):
            mask = folds[:, 0] == folds[:, 0][i]
            train_dist[i, mask] = np.nan
        end = time.time()
        print(f"Masking took {end - start} seconds")

    print("Calculating AOA...")
    start = time.time()
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
    end = time.time()
    print(f"Calculating AOA took {end - start} seconds")

    return DIs, masked_result
