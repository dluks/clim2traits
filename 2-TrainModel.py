# # 2: Train XGBoost Model
#
# Author: Daniel Lusk

# ## Imports and configuration

import argparse
from pathlib import Path
from pprint import pprint
from typing import Optional

import geopandas as gpd

from TrainModelConfig import TrainModelConfig
from utils.dataset_tools import FileExt, Unit
from utils.datasets import (
    CollectionName,
    DataCollection,
    Dataset,
    GBIFBand,
    MLCollection,
)

config = TrainModelConfig()

# ## Load data


# %%
def prep_data(
    X_names: list = ["all"],
    Y_names: list = ["inat_gbif"],
    res: float = 0.5,
    X_collection: Optional[str] = None,
) -> MLCollection:
    """Load data and prepare for training"""

    # Prep Y data
    gbif = Dataset(
        res=res,
        unit=Unit.DEGREE,
        collection_name=CollectionName.GBIF,
        band=GBIFBand.MEAN,
        file_ext=FileExt.GRID,
    )

    splot = Dataset(
        res=res,
        unit=Unit.DEGREE,
        collection_name=CollectionName.SPLOT,
        band=GBIFBand.MEAN,
        file_ext=FileExt.GRID,
    )

    all_rvs = [gbif, splot]

    if Y_names == ["all"]:
        rvs = all_rvs
    else:
        rvs: list[Dataset] = []
        for rv in all_rvs:
            if rv.collection_name.abbr in Y_names:
                rvs.append(rv)

    Y = DataCollection(rvs)

    # Prep X data
    if X_collection is not None:
        print("\nUsing collection: ", Path(X_collection).name)
        df = gpd.read_feather(X_collection)
        X = DataCollection.from_df(df)
    else:
        wc = Dataset(
            res=res,
            unit=Unit.DEGREE,
            collection_name=config.WC_name,
        )

        modis = Dataset(
            res=res,
            unit=Unit.DEGREE,
            collection_name=config.MODIS_name,
        )

        soil = Dataset(
            res=res,
            unit=Unit.DEGREE,
            collection_name=config.soil_name,
        )

        vodca = Dataset(
            res=0.5,
            unit=Unit.DEGREE,
            collection_name=CollectionName.VODCA,
            file_ext=FileExt.NETCDF4,
        )

        all_predictors = [wc, modis, soil, vodca]

        if X_names == ["all"]:
            predictors = all_predictors
        else:
            predictors: list[Dataset] = []
            for predictor in all_predictors:
                if predictor.collection_name.abbr in X_names:
                    predictors.append(predictor)

        X = DataCollection(predictors)

    print("\nPreparing data...")
    print("X:")
    for dataset in X.datasets:
        print("    ", dataset.collection_name.short)

    print("Y:")
    for dataset in Y.datasets:
        print("    ", dataset.collection_name.short)

    # Convert to MLCollection for training
    XY = MLCollection(X, Y)
    XY.drop_NAs(verbose=1)

    return XY


# ### Train models for each response variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", nargs="+", type=str, default=["all"], help="X datasets")
    parser.add_argument("--Y", nargs="+", type=str, default=["gbif"], help="Y datasets")
    parser.add_argument(
        "--res", type=float, default=0.5, help="Resolution of the datasets"
    )
    parser.add_argument(
        "--inat_transform",
        type=str,
        default="exp_ln",
        help="iNaturalist transform to use as response variable",
        choices=["exp_ln", "ln"],
    )
    parser.add_argument(
        "--X-collection",
        type=str,
        default=None,
        help="Path to existing X collection dataframe.",
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

    args = parser.parse_args()

    config = TrainModelConfig(debug=args.debug)

    if not args.filter_outliers:
        config.training_config.filter_y_outliers = None

    if args.debug:
        print(f"X", args.X)
        print(f"Y", args.Y)
        print(f"Resolution: {args.res}")
        print(f"iNaturalist transform: {args.inat_transform}")
        print(f"Resume: {args.resume}")
        print(f"Y indices: {args.y_idx}")
        print(f"Debug: {args.debug}")
        print(f"Config:")
        pprint(config.__dict__)

    XY = prep_data(
        X_names=args.X, Y_names=args.Y, res=args.res, X_collection=args.X_collection
    )

    if args.debug:
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
