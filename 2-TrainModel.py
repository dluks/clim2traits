#!/usr/bin/env python3
# %% [markdown]
# # 2: Train XGBoost Model
#
# Author: Daniel Lusk

# %% [markdown]
# ## Imports and configuration

# %%

import numpy as np
import spacv
from spacv.visualisation import plot_autocorrelation_ranges

from TrainModelConfig import TrainModelConfig
from utils.datasets import DataCollection, Dataset, MLCollection, Unit
from utils.geodata import drop_XY_NAs
from utils.visualize import plot_splits

# NOTEBOOK = False

# if NOTEBOOK:
#     %load_ext autoreload
#     %autoreload 2

config = TrainModelConfig()

# %% [markdown]
# ## Load data

# %%
inat = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    parent_dir=config.iNat_dir,
    collection_name=config.iNat_name,
    transform="ln",
)

wc = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    parent_dir=config.WC_dir,
    collection_name=config.WC_name,
    bio_ids=config.WC_bio_ids,
)

modis = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    parent_dir=config.MODIS_dir,
    collection_name=config.MODIS_name,
)

soil = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    parent_dir=config.soil_dir,
    collection_name=config.soil_name,
)

X = DataCollection([wc, modis, soil])
Y = DataCollection([inat])

# Convert to MLCollection for training
XY = MLCollection(X, Y)
print("XY shape:", XY.df.shape)

XY.drop_NAs(verbose=1)

# %% [markdown]
# ## XGBoost

# %% [markdown]
# <div class="alert alert-block alert-info">
# To-Dos:
#
# 1) ~~Create a data frame where you have all response variables and predictors.~~
# 2) ~~Remove cells where you do not have a value for ANY predictor/response variable (you still may have NA for some columns then).~~
# 3) ~~Train the models and do the evaluation~~
# 4) Repeat step 3, but remove rows where you have at least one NA
# 5) Compare accuracies of step 3 and 4 and see what´s best.
# </div>

# %% [markdown]
# ### Calculate autocorrelation range of predictors and generate spatial folds for spatial cross-validation

# %%
if config.SAVE_AUTOCORRELATION_RANGES:
    coords = XY["geometry"]
    data = XY[X.cols]

    _, _, ranges = plot_autocorrelation_ranges(
        coords, data, config.LAGS, config.BW, distance_metric="haversine", workers=10
    )

    np.save("ranges.npy", np.asarray(ranges))

# %% [markdown]
# #### Explore splits for a single response variable

# %%
if config.EXPLORE_SPLITS:
    y_col = "iNat_Stem.conduit.density_05deg_ln"
    sample_Xy = XY.df[["geometry", *XY.X.cols, y_col]]

    # Drop full-NAs
    sample_Xy, sample_X_cols, sample_y_col = drop_XY_NAs(
        sample_Xy, XY.X.cols, y_col, True
    )

    # Sample X data on which split dissimilarity will be measured
    sample_data = sample_Xy[sample_X_cols]
    sample_locs = sample_Xy["geometry"]

    # Grid settings
    tile = config.AUTOCORRELATION_RANGE / config.DEGREE
    tiles_x = int(np.round(360 / tile))
    tiles_y = int(np.round(180 / tile))

    # Spatial blocking
    hblock = spacv.HBLOCK(
        tiles_x,
        tiles_y,
        shape="hex",
        method="optimized_random",
        buffer_radius=0.01,
        n_groups=10,
        data=sample_data,
        n_sims=50,
        distance_metric="haversine",
        random_state=config.RNG_STATE,
    )

    # Plot splits
    print(f"Tile size: {tile:.2f} degrees")
    plot_splits(hblock, sample_locs)

# %% [markdown]
# ### Train models for each response variable

# %%
if config.TRAIN_MODE:
    XY.train_collection(config.training_config)

# %% [markdown]
# # Debugging

# %%
if config.DEBUG:
    import pathlib

    from utils.training import TrainingConfig, block_cv_splits, optimize_params

    results_dir = pathlib.Path(config.RESULTS_DIR, "test")
    results_csv = pathlib.Path(results_dir, "training_results.csv")
    train_config = TrainingConfig(
        cv_n_groups=2,
        search_n_trials=2,
        results_dir=results_dir,
        results_csv=results_csv,
    )

    y_col = XY.Y.cols[0]

    Xy = XY.df[["geometry", *XY.X.cols, y_col]]
    Xy, X_cols, y_cols = drop_XY_NAs(Xy, XY.X.cols, y_col, True)

    X = Xy[X_cols].to_numpy()
    y = Xy[y_col].to_numpy()
    coords = Xy["geometry"]

    cv = block_cv_splits(
        X=X,
        coords=coords,
        grid_size=train_config.cv_grid_size,
        n_groups=train_config.cv_n_groups,
        random_state=train_config.random_state,
        verbose=1,
    )

    reg = optimize_params(
        X=X,
        y=y,
        col_name=y_col,
        cv=cv,
        save_dir=train_config.results_dir,
        n_trials=train_config.search_n_trials,
        random_state=train_config.random_state,
    )
