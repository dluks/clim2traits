# # 2: Train XGBoost Model
#
# Author: Daniel Lusk

# ## Imports and configuration

import argparse
from pathlib import Path
from pprint import pprint
from typing import Optional, Union

from scripts.save_collection import (
    build_collection,
    build_collection_fn,
    save_collection,
)
from TrainModelConfig import TrainModelConfig
from utils.dataset_tools import Unit
from utils.datasets import (
    CollectionName,
    DataCollection,
    Dataset,
    GBIFBand,
    MLCollection,
)

config = TrainModelConfig()


def get_all_rvs(resolution: Union[float, int], pft: str) -> list[Dataset]:
    """Get all response variables as a list of Datasets"""
    gbif = Dataset(
        res=resolution,
        unit=Unit.DEGREE,
        collection_name=CollectionName.GBIF,
        band=GBIFBand.MEAN,
        pft=pft,
    )

    splot = Dataset(
        res=resolution,
        unit=Unit.DEGREE,
        collection_name=CollectionName.SPLOT,
        band=GBIFBand.MEAN,
        pft=pft,
    )

    gbif_ln = Dataset(
        res=resolution,
        collection_name=CollectionName.GBIF_LN,
        pft=pft,
    )

    splot_ln = Dataset(
        res=resolution,
        collection_name=CollectionName.SPLOT_LN,
        pft=pft,
    )

    return [gbif, splot, gbif_ln, splot_ln]


def get_all_predictors(resolution: Union[float, int]) -> list[Dataset]:
    """Get all predictors as a list of Datasets"""
    wc = Dataset(
        res=resolution,
        unit=Unit.DEGREE,
        collection_name=config.WC_name,
    )

    modis = Dataset(
        res=resolution,
        unit=Unit.DEGREE,
        collection_name=config.MODIS_name,
    )

    soil = Dataset(
        res=resolution,
        unit=Unit.DEGREE,
        collection_name=config.soil_name,
    )

    vodca = Dataset(
        res=resolution,
        unit=Unit.DEGREE,
        collection_name=CollectionName.VODCA,
    )

    return [wc, modis, soil, vodca]


def get_specific_datasets(
    ds_names: list[str], all_datasets: list[Dataset]
) -> list[Dataset]:
    """Get specific datasets from a list of all datasets"""
    datasets: list[Dataset] = []
    for ds in all_datasets:
        if ds.collection_name.abbr in ds_names:
            datasets.append(ds)

    if len(datasets) == 0:
        raise ValueError(f"Could not find datasets: {ds_names}")

    return datasets


def prep_data(
    X_names: list,
    Y_names: list,
    res: float = 0.5,
    pft: str = "Shrub_Tree_Grass",
    nan_strategy: Optional[str] = None,
    thresh: Optional[float] = None,
    X_collection: Optional[str] = None,
) -> MLCollection:
    """Load data and prepare for training"""

    if X_names is None:
        X_names = ["all"]

    if Y_names is None:
        Y_names = ["all"]

    # Reponse variables
    all_rvs = get_all_rvs(res, pft)

    if Y_names == ["all"]:
        rvs = all_rvs
    else:
        rvs = get_specific_datasets(Y_names, all_rvs)

    Y = DataCollection(rvs)

    # Predictors
    if X_collection is not None:
        print("\nUsing collection: ", Path(X_collection).name)
        X = DataCollection.from_collection(X_collection)
    else:
        all_predictors = get_all_predictors(res)

        if X_names == ["all"]:
            predictors = all_predictors
        else:
            predictors = get_specific_datasets(X_names, all_predictors)

        predictor_names = [ds.collection_name.abbr for ds in predictors]

        X = DataCollection(predictors)

        X, coll_name = build_collection(
            data_collection=X,
            res=res,
            nan_strategy=nan_strategy,
            predictor_names=predictor_names,
            thresh=thresh,
        )

        coll_fn = build_collection_fn(coll_name)

        if Path(coll_fn).exists():
            print("Collection already exists. Loading...")
            X = DataCollection.from_collection(coll_fn)
        else:
            print("Saving collection...")
            save_collection(X, coll_name, impute=False)
            X.collection_file = coll_fn

    print("\nPreparing data...")
    print("X:")
    for x_ds in X.datasets:
        print("    ", x_ds.collection_name.short)

    print("Y:")
    for y_ds in Y.datasets:
        print("    ", y_ds.collection_name.short)

    # Convert to MLCollection for training
    XY_collection = MLCollection(X, Y)
    XY_collection.drop_NAs(verbose=1)

    return XY_collection


# ### Train models for each response variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", nargs="+", type=str, default=["all"], help="X datasets")
    parser.add_argument("--Y", nargs="+", type=str, default=["all"], help="Y datasets")
    parser.add_argument(
        "--res", type=float, default=0.5, help="Resolution of the datasets"
    )
    parser.add_argument("--pft", type=str, default="Shrub_Tree_Grass", help="PFT")
    parser.add_argument(
        "--X-collection",
        type=str,
        default=None,
        help="Path to existing X collection dataframe.",
    )
    parser.add_argument("--nan-strategy", type=str, default=None, help="NaN strategy")
    parser.add_argument(
        "--thresh",
        type=float,
        default=None,
        help="Filter rows with NaNs exceeding this threshold.",
    )
    parser.add_argument("--tune", action="store_true", help="Tune hyperparameters")
    parser.add_argument(
        "--resume", action="store_true", help="Resume previous training run"
    )
    parser.add_argument(
        "--y_idx",
        nargs="+",
        type=int,
        default=None,
        help="Y indices to train. Usage: `--y_idx 0 3 23 24`. If None, train all",
    )
    parser.add_argument(
        "--filter-outliers", action="store_true", help="Filter out Y outliers"
    )
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    parser.add_argument("--verbose", action="store_true", help="Verbose mode")

    args = parser.parse_args()

    config = TrainModelConfig(debug=args.debug)

    if not args.filter_outliers:
        config.training_config.filter_y_outliers = None

    if args.verbose or args.debug:
        print("X", args.X)
        print("Y", args.Y)
        print(f"Resolution: {args.res}")
        print(f"PFT: {args.pft}")
        print(f"Resume: {args.resume}")
        print(f"Y indices: {args.y_idx}")
        print(f"Debug: {args.debug}")
        print("Config:")
        pprint(config.__dict__)

    XY = prep_data(
        X_names=args.X,
        Y_names=args.Y,
        res=args.res,
        pft=args.pft,
        nan_strategy=args.nan_strategy,
        thresh=args.thresh,
        X_collection=args.X_collection,
    )

    if args.verbose or args.debug:
        print("X datasets:")
        for dataset in XY.X.datasets:
            print(dataset.collection_name)

        print("Y datasets:")
        for dataset in XY.Y.datasets:
            print(dataset.collection_name)

    XY.train_Y_models(
        training_config=config.training_config,
        y_idx=args.y_idx,
        tune_params=args.tune,
        resume=args.resume,
    )
