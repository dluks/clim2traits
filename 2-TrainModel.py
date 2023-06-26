# %% [markdown]
# # 2: Train XGBoost Model
#
# Author: Daniel Lusk

# %% [markdown]
# ## Imports and configuration

import argparse
import pathlib

from TrainModelConfig import TrainModelConfig
from utils.datasets import CollectionName, DataCollection, Dataset, MLCollection, Unit

config = TrainModelConfig()

# %% [markdown]
# ## Load data


# %%
def prep_data(
    X_names: list = ["wc", "soil", "modis"],
    Y_names: list = ["inat_orig, inat_dgvm"],
    res: float = 0.5,
    inat_transform: str = "exp_ln",
) -> MLCollection:
    inat_orig = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=config.iNat_dir,
        collection_name=config.iNat_name,
        transform=inat_transform,
    )

    inat_dgvm = Dataset(
        res=res,
        unit=Unit.DEGREE,
        parent_dir=pathlib.Path(
            "./iNaturalist_traits/maps_iNaturalist/DGVM/continuous_traits/"
        ),
        collection_name=CollectionName.INAT_DGVM,
        transform=inat_transform,
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

    all_predictors = [wc, modis, soil]
    all_rvs = [inat_orig, inat_dgvm]

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

    # Convert to MLCollection for training
    XY = MLCollection(X, Y)
    print("XY shape:", XY.df.shape)

    XY.drop_NAs(verbose=1)

    return XY


# ### Train models for each response variable

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--X", nargs="+", type=str, default=["all"], help="X datasets")
    parser.add_argument("--Y", nargs="+", type=str, default=["all"], help="Y datasets")
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
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    config = TrainModelConfig(debug=args.debug)

    if args.debug:
        print(f"X", args.X)
        print(f"Y", args.Y)
        print(f"Resolution: {args.res}")
        print(f"iNaturalist transform: {args.inat_transform}")
        print(f"Resume: {args.resume}")
        print(f"Y indices: {args.y_idx}")
        print(f"Debug: {args.debug}")
        print(f"Config: {config.__dict__}")

    XY = prep_data(
        X_names=args.X, Y_names=args.Y, res=args.res, inat_transform=args.inat_transform
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