# # 2: Train XGBoost Model
#
# Author: Daniel Lusk

# ## Imports and configuration

import argparse
from pathlib import Path
from pprint import pprint

from TrainModelConfig import TrainModelConfig
from utils.dataset_tools import FileExt, Unit
from utils.datasets import CollectionName, DataCollection, Dataset, MLCollection

config = TrainModelConfig()

# ## Load data


# %%
def prep_data(
    X_names: list = ["all"],
    Y_names: list = ["inat_gbif"],
    res: float = 0.5,
    y_transform: str = "exp_ln",
) -> MLCollection:
    """Load data and prepare for training"""
    inat_orig = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=config.iNat_dir,
        collection_name=config.iNat_name,
        transform=y_transform,
    )

    inat_dgvm = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=Path(
            "./iNaturalist_traits/maps_iNaturalist/DGVM/continuous_traits/"
        ),
        collection_name=CollectionName.INAT_DGVM,
        transform=y_transform,
    )

    inat_gbif = Dataset(
        res=0.5,
        unit=Unit.DEGREE,
        parent_dir=Path("./iNaturalist_traits/maps_GBIF/traitmaps/TRY_gap_filled/"),
        collection_name=CollectionName.INAT_GBIF,
        filter_outliers=config.training_config.filter_y_outliers,
    )

    splot = Dataset(
        res=0.5,
        unit=Unit.DEGREE,
        parent_dir=config.splot_dir,
        collection_name=CollectionName.SPLOT,
        transform=y_transform,
    )

    wc = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=config.WC_dir,
        collection_name=config.WC_name,
        bio_ids=config.WC_bio_ids,
    )

    modis = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=config.MODIS_dir,
        collection_name=config.MODIS_name,
    )

    soil = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=config.soil_dir,
        collection_name=config.soil_name,
    )

    vodca = Dataset(
        res=0.5,
        unit=Unit.DEGREE,
        parent_dir=Path("./data/vodca/"),
        collection_name=CollectionName.VODCA,
        file_ext=FileExt.NETCDF4,
    )

    all_predictors = [wc, modis, soil, vodca]
    all_rvs = [inat_orig, inat_dgvm, inat_gbif, splot]

    if X_names == ["all"]:
        predictors = all_predictors
    else:
        predictors: list[Dataset] = []
        for predictor in all_predictors:
            if predictor.collection_name.abbr in X_names:
                predictors.append(predictor)

    if Y_names == ["all"]:
        rvs = all_rvs
    else:
        rvs: list[Dataset] = []
        for rv in all_rvs:
            if rv.collection_name.abbr in Y_names:
                rvs.append(rv)

    X = DataCollection(predictors)
    Y = DataCollection(rvs)

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
    parser.add_argument(
        "--Y", nargs="+", type=str, default=["inat_gbif"], help="Y datasets"
    )
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
        X_names=args.X, Y_names=args.Y, res=args.res, y_transform=args.inat_transform
    )

    if args.debug:
        print("X datasets:")
        for dataset in XY.X.datasets:
            print(dataset.collection_name)

        print("Y datasets:")
        for dataset in XY.Y.datasets:
            print(dataset.collection_name)

    XY.train_Y_models(
        training_config=config.training_config, y_idx=args.y_idx, resume=args.resume
    )
