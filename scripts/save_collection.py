#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.experimental import enable_iterative_imputer  # type: ignore
from sklearn.impute import IterativeImputer

from utils.datasets import DataCollection


def impute(
    data: np.ndarray,
    n_iter: int = 10,
    fitted_imputer: Optional[IterativeImputer] = None,
    resume_fit: bool = False,
) -> Tuple[np.ndarray, IterativeImputer]:
    """Impute missing values using iterative imputer

    Args:
        data (np.ndarray): Data with missing values
        n_iter (int, optional): Number of iterations. Defaults to 10.

    Returns:
        np.ndarray: Imputed data
    """
    if fitted_imputer is not None:
        imp = fitted_imputer

        if resume_fit:
            imp.fit(data)
    else:
        imp = IterativeImputer(max_iter=n_iter, random_state=42, verbose=2)
        imp.fit(data)

    data_imputed = imp.transform(data)

    return data_imputed, imp


def build_collection(
    res_str: str = "0.5_deg", nan_strategy: str = "all", thresh: Optional[float] = None
) -> Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], str]:
    valid_nan_strategies = ["all", "any"]
    if nan_strategy not in valid_nan_strategies:
        raise ValueError(
            f"Invalid nan_strategy. Valid options are {valid_nan_strategies}"
        )

    if nan_strategy == "all" and thresh is not None:
        warnings.warn(
            "Cannot specify thresh when nan_strategy is 'all'. Setting to None."
        )
        thresh = None

    predictor_names = ["MOD09GA.061", "ISRIC_soil", "WC_BIO", "VODCA"]
    # predictor_names = ["MOD09GA.061"]
    predictor_ids = [f"{id}_{res_str}" for id in predictor_names]

    X = DataCollection.from_ids(predictor_ids)

    # # Mask non-land points
    # print("Masking non-land points")
    # land_mask = gpd.read_feather("./data/masks/land_mask_110m.feather")
    # X.df = dgpd.clip(X.df, land_mask)

    # print("Dropping NaN columns from X")
    # X.df = X.df.dropna(axis=1, how="all")  # Drop any all-NaN columns
    # Drop any all-NaN columns (this is slow and ugly but necessary due to limitations
    # in dask_geopandas)
    # start = time.time()
    # X.df = X.df.loc[:, ~X.df.isna().all().compute()]
    # end = time.time()
    # print(f"Took: {(end - start) / 60:.2f} minutes")

    print("Getting X cols")
    X_cols = X.cols

    print("Dropping NaN rows from X")
    if nan_strategy == "all":
        X.df = X.df.dropna(subset=X_cols, how="all")
    elif nan_strategy == "any":
        if thresh is None:
            X.df = X.df.dropna(subset=X_cols, how="any")
        else:
            X.df = X.df.dropna(subset=X_cols, thresh=int(thresh * len(X_cols)))

    collection_name = "_".join(predictor_names)
    collection_name += f"_{res_str}"
    collection_name += f"_nan-strat={nan_strategy}"
    if thresh is not None:
        collection_name += f"_thr={thresh}"

    return X.df, collection_name


def save_collection(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    collection_name: str,
    impute_missing: bool = False,
    n_iter: int = 10,
    fitted_imputer: Optional[IterativeImputer] = None,
    resume_fit: bool = False,
    save_imputer: bool = False,
) -> None:
    if impute_missing:
        print("Imputing missing values")

        data_cols = df.columns.difference(["geometry"])

        if isinstance(df, gpd.GeoDataFrame):
            # extract x and y from geometry
            x, y = df.geometry.x, df.geometry.y
            data = np.c_[df[data_cols].to_numpy(), x, y]
        else:
            data = df.to_numpy()

        data, imputer = impute(
            data, n_iter=n_iter, fitted_imputer=fitted_imputer, resume_fit=resume_fit
        )

        if fitted_imputer is None and save_imputer:
            out_dir = Path("models/imputers")
            out_dir.mkdir(exist_ok=True, parents=True)
            out_fn = out_dir / f"imputer_{collection_name}.joblib"
            dump(imputer, out_fn)

        if isinstance(df, gpd.GeoDataFrame):
            geometry = df.pop("geometry")
            data = data[:, :-2]  # drop x and y (only needed for imputing)
            df = gpd.GeoDataFrame(
                data, columns=data_cols, geometry=geometry, index=df.index
            )
        else:
            df = pd.DataFrame(data, columns=df.columns)

    if impute_missing:
        out_stem = f"{collection_name}_imputed"
    else:
        out_stem = collection_name

    if isinstance(
        df, (gpd.GeoDataFrame, dgpd.GeoDataFrame, pd.DataFrame, dd.DataFrame)
    ):
        print("Saving collection...")
        out_dir = Path("data/collections")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_fn = out_dir / f"{out_stem}.parquet"
        df.to_parquet(out_fn)
    else:
        raise TypeError("df must be a GeoDataFrame or DataFrame")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res-str",
        type=str,
        default="0.5_deg",
        help="Resolution string. E.g. '0.5_deg'",
    )
    parser.add_argument(
        "--nan-strategy",
        type=str,
        default="all",
        help="Strategy for handling NaNs. Options are 'all' or 'any'",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Threshold for number of NaNs in a row. E.g. 0.95 would remove rows that have > 95% NaNs",
    )
    parser.add_argument(
        "--impute-missing", action="store_true", help="Whether to impute missing values"
    )
    parser.add_argument(
        "--n-iter", type=int, default=20, help="Number of iterations for imputation"
    )
    parser.add_argument(
        "--fitted-imputer", type=str, default=None, help="Path to fitted imputer"
    )
    parser.add_argument(
        "--resume-fit", action="store_true", help="Whether to resume fitting imputer"
    )
    parser.add_argument(
        "--save-imputer", action="store_true", help="Whether to save fitted imputer"
    )
    parser.add_argument(
        "--collection", type=str, default=None, help="Path to existing collection"
    )

    args = parser.parse_args()

    if args.fitted_imputer is not None:
        print(f"Loading fitted imputer from {args.fitted_imputer}")
        args.fitted_imputer = load(args.fitted_imputer)

    print("Building collection")
    if args.collection is not None:
        print("Loading existing collection")
        collection_path = Path(args.collection)
        if collection_path.suffix == ".parquet":
            df = gpd.read_parquet(args.collection)
        elif collection_path.suffix == ".feather":
            df = gpd.read_feather(args.collection)
        collection_name = collection_path.stem
    else:
        df, collection_name = build_collection(
            res_str=args.res_str, nan_strategy=args.nan_strategy, thresh=args.thresh
        )

    save_collection(
        df,
        collection_name,
        impute_missing=args.impute_missing,
        n_iter=args.n_iter,
        fitted_imputer=args.fitted_imputer,
        resume_fit=args.resume_fit,
        save_imputer=args.save_imputer,
    )
    print("Saving complete")
