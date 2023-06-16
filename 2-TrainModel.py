#!/usr/bin/env python3
# %% [markdown]
# # 2: Train XGBoost Model
#
# Author: Daniel Lusk

# %% [markdown]
# ## Imports and configuration

import os
from datetime import datetime

# %%
import numpy as np
import pandas as pd
from spacv.visualisation import plot_autocorrelation_ranges

from TrainModelConfig import TrainModelConfig
from utils.data_retrieval import gdf_from_list
from utils.geodata import drop_NAs, merge_gdfs
from utils.training import run_training

config = TrainModelConfig()

# %% [markdown]
# ## Load data

# %%
X_fns = config.WC_fns + config.MODIS_fns + config.soil_fns
Y_fns = config.iNat_fns

X = gdf_from_list(X_fns)
Y = gdf_from_list(Y_fns)

# %% [markdown]
# Compute Preciptation Annual Range by subtracting BIO14 from BIO13

# %%
bio_13 = X.loc[:, ["bio_13" in x for x in X.columns]].values
bio_14 = X.loc[:, ["bio_14" in x for x in X.columns]].values
X["wc2.1_10m_bio_13-14"] = bio_13 - bio_14

# %% [markdown]
# Drop the unnecessary `x`, `y`, `band` and `spatial_ref` columns.

# %%
X = X.drop(columns=["x", "y", "band", "spatial_ref"])
Y = Y.drop(columns=["x", "y", "band", "spatial_ref"])

# %% [markdown]
# ## XGBoost

# %% [markdown]
# <div class="alert alert-block alert-info">
# To-Dos:
#
# 1) ~~Create a data frame where you have all response variables and predictors.~~
# 2) ~~Remove cells where you do not have a value for ANY predictor/response variable (you still may have NA for some columns then).~~
# 3) Train the models and do the evaluation
# 4) Repeat step 3, but remove rows where you have at least one NA
# 5) Compare accuracies of step 3 and 4 and see what´s best.
# </div>

# %% [markdown]
# ### Combine GDFs and clean up nodata

# %% [markdown]
# Get column names for easier predictor/response variable selection.

# %%
X_cols = X.columns.difference(["geometry"])
Y_cols = Y.columns.difference(["geometry"])

# %% [markdown]
# Merge X and Y GDFs

# %%
XY = merge_gdfs([X, Y])
print("X shape:", XY[X_cols].shape)
print("Y shape:", XY[Y_cols].shape)

# %% [markdown]
# Drop all-NA rows and columns

# %%
XY, X_cols, Y_cols = drop_NAs(XY, X_cols, Y_cols, True)

# %% [markdown]
# ### Calculate autocorrelation range of predictors and generate spatial folds for spatial cross-validation

# %%
if config.SAVE_AUTOCORRELATION_RANGES:
    coords = XY["geometry"]
    data = XY[X_cols]

    _, _, ranges = plot_autocorrelation_ranges(
        coords, data, config.LAGS, config.BW, distance_metric="haversine", workers=10
    )

    np.save("ranges.npy", np.asarray(ranges))

# %% [markdown]
# ### Train models for each response variable
# %%
if __name__ == "__main__":
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    param_opt_run_dir = os.path.join(config.PARAM_OPT_RESULTS_DIR, run_time)
    results_dir = os.path.join(config.MODEL_DIR, run_time)

    if not os.path.exists(param_opt_run_dir):
        os.makedirs(param_opt_run_dir)

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_fn = os.path.join(results_dir, f"{run_time}_results.csv")
    data_cols = ["model", "params", "mean rmse", "std", "r-squared"]
    results_df = pd.DataFrame(columns=data_cols)

    for y_col in Y_cols[-1:]:
        print(f"Processing {y_col}...\n")
        run_name = f"{run_time}_{y_col}"

        Xy = XY[["geometry", *X_cols, y_col]]

        Xy, X_cols, y_col = drop_NAs(Xy, X_cols, y_col, True)

        model_fn, params, rmse, std, r2 = run_training(
            Xy=Xy,
            X_cols=X_cols,
            y_col=y_col,
            autocorr_range=config.AUTOCORRELATION_RANGE,
            search_n_trials=100,
            n_jobs=-1,
            random_state=config.RNG_STATE,
            param_opt_save_dir=param_opt_run_dir,
            final_save_dir=results_dir,
            run_name=run_name,
        )
        results = [model_fn, params, rmse, std, r2]
        new_df = pd.DataFrame([results], columns=data_cols)
        results_df = pd.concat([results_df, new_df])

        results_df.to_csv(os.path.join(config.MODEL_DIR, results_fn))
