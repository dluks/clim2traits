#!/usr/bin/env python3

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import dask.dataframe as dd
import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # type: ignore

from utils.datasets import DataCollection
from utils.geodata import num_to_str
from utils.spatial_stats import impute_missing


def build_collection(
    data_collection: Optional[DataCollection] = None,
    res: Union[float, int] = 0.5,
    predictor_names: Optional[list[str]] = None,
    nan_strategy: Optional[str] = None,
    thresh: Optional[float] = None,
) -> Tuple[DataCollection, str]:
    """Build a collection of predictors. If data_collection is not provided, then one
    will be built from the res."""
    valid_nan_strategies = ["all", "any", None]
    if nan_strategy not in valid_nan_strategies:
        raise ValueError(
            f"Invalid nan_strategy. Valid options are {valid_nan_strategies}"
        )

    if nan_strategy in ["all", "any"] and thresh is not None:
        raise ValueError("Cannot specify thresh when nan_strategy is 'all' or 'any'.")

    if nan_strategy is None and thresh is None:
        nan_strategy = "all"

    if data_collection is None:
        if predictor_names is None:
            predictor_names = ["MOD09GA.061", "ISRIC_soil", "WC_BIO", "VODCA"]
        predictor_ids = [f"{id}_{res:g}_deg" for id in predictor_names]
        data_collection = DataCollection.from_ids(predictor_ids)
    else:
        predictor_names = (
            [ds.collection_name.abbr for ds in data_collection.datasets]
            if predictor_names is None
            else predictor_names
        )

    logging.info("Getting X cols")
    X_cols = data_collection.cols

    logging.info("Dropping NaN rows from X")
    if nan_strategy is not None:
        data_collection.df = data_collection.df.dropna(subset=X_cols, how=nan_strategy)
    else:
        data_collection.df = data_collection.df.dropna(
            subset=X_cols, thresh=int(thresh * len(X_cols))
        )

    coll_name = "_".join(predictor_names)
    coll_name += f"_{num_to_str(res)}deg"
    coll_name += f"_nan-strat={nan_strategy}"
    if thresh is not None:
        coll_name += f"_thr={thresh}"

    return data_collection, coll_name
    # return data_collection.df.reset_index(drop=True), coll_name


def build_collection_fn(collection_name: str, impute: bool = False) -> str:
    """Build a collection from a collection name."""
    parent_dir = Path("data/collections")
    parent_dir.mkdir(exist_ok=True, parents=True)
    fn = f"{collection_name}{'_imputed' if impute else ''}.parquet"
    return str(parent_dir / fn)


def save_collection(
    collection: DataCollection,
    collection_name: str,
    impute: bool = False,
) -> None:
    """Save a collection to disk."""
    df = collection.df.reset_index(drop=True)

    if impute:
        logging.info("Imputing missing values")

        data_cols = df.columns.difference(["geometry"])

        df = df.reset_index(drop=True)
        geom = df.geometry

        df_out = impute_missing(df[data_cols], verbose=True)

        df_out = gpd.GeoDataFrame(
            df_out.reset_index(drop=True),
            geometry=geom,
            index=df.index,
        )
    else:
        df_out = df

    if isinstance(
        df_out, (gpd.GeoDataFrame, dgpd.GeoDataFrame, pd.DataFrame, dd.DataFrame)
    ):
        logging.info("Saving collection...")
        out_fn = build_collection_fn(collection_name, impute)
        df_out.to_parquet(out_fn, compression="zstd", compression_level=2)
    else:
        raise TypeError("df must be a GeoDataFrame or DataFrame")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--res",
        type=float,
        default=0.5,
        help="Resolution in degrees. Defaults to 0.5.",
    )
    parser.add_argument(
        "--nan-strategy",
        type=str,
        default=None,
        help="Strategy for handling NaNs. Options are 'all' or 'any'. Defaults to None.",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Threshold for number of NaNs in a row. E.g. 0.95 would remove rows that"
        "have > 95% NaNs",
    )
    parser.add_argument(
        "--impute-missing", action="store_true", help="Whether to impute missing values"
    )
    parser.add_argument(
        "--collection", type=str, default=None, help="Path to existing collection"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    logging.info("Building collection")
    if args.collection is not None:
        logging.info("Loading existing collection")
        collection_path = Path(args.collection)
        if collection_path.suffix == ".parquet":
            df = gpd.read_parquet(args.collection)
        elif collection_path.suffix == ".feather":
            df = gpd.read_feather(args.collection)
        collection_name = collection_path.stem
    else:
        df, collection_name = build_collection(
            res=args.res, nan_strategy=args.nan_strategy, thresh=args.thresh
        )

    save_collection(
        df,
        collection_name,
        impute=args.impute_missing,
    )
    logging.info("Saving complete")


if __name__ == "__main__":
    main()
