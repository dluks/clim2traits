#!/usr/bin/env python3

import argparse
import warnings
from pathlib import Path
from typing import Optional, Tuple, Union

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # type: ignore

from utils.datasets import DataCollection
from utils.spatial_stats import impute_missing


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

    return X.df.reset_index(drop=True), collection_name


def save_collection(
    df: Union[gpd.GeoDataFrame, pd.DataFrame],
    collection_name: str,
    impute: bool = False,
) -> None:
    if impute:
        print("Imputing missing values")

        data_cols = df.columns.difference(["geometry"])

        # extract x and y from geometry and append as columns
        # df["x"] = df.geometry.x
        # df["y"] = df.geometry.y
        df = df.reset_index(drop=True)
        geom = df.geometry

        df_out = impute_missing(df[data_cols], verbose=True)

        df_out = gpd.GeoDataFrame(
            df_out.reset_index(drop=True),
            geometry=geom,
            index=df.index,
        )
        out_stem = f"{collection_name}_imputed"
    else:
        out_stem = collection_name
        df_out = df

    if isinstance(
        df_out, (gpd.GeoDataFrame, dgpd.GeoDataFrame, pd.DataFrame, dd.DataFrame)
    ):
        print("Saving collection...")
        out_dir = Path("data/collections")
        out_dir.mkdir(exist_ok=True, parents=True)
        out_fn = out_dir / f"{out_stem}.parquet"
        df_out.to_parquet(out_fn, compression="zstd", compression_level=2)
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
        "--collection", type=str, default=None, help="Path to existing collection"
    )

    args = parser.parse_args()

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
        impute=args.impute_missing,
    )
    print("Saving complete")
