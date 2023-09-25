import argparse
import json
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd

from utils.models import Prediction, TrainedSet

results = pd.read_csv("./results/training_results.csv").copy()

with open("trait_id_to_trait_name.json", "r") as f:
    trait_map = json.load(f)

# First filter results to only include rows that were not imputed. Only get rows where
# "NaN strategy" does not contain "imputed"
results = results.loc[~results["NaN strategy"].str.contains("imputed")]
# results = results.loc[results["NaN strategy"] == "threshold-0.5"]

# Filter results to only include rows in which Response variable starts with GBIF
gbif_res = results.loc[results["Response variable"].str.startswith("GBIF")]
splot_res = results.loc[results["Response variable"].str.startswith("sPlot")]

# Filter results to only include rows in which Run ID is one of the most recent 4 run IDs
gbif_res = gbif_res.loc[gbif_res["Run ID"].isin(gbif_res["Run ID"].unique()[-2:])]
splot_res = splot_res.loc[splot_res["Run ID"].isin(splot_res["Run ID"].unique()[-2:])]

# Filter results to only include rows in which Response variable does not contain "_ln"
# gbif_res = gbif_res.loc[~gbif_res["Response variable"].str.contains("_ln")]
# splot_res = splot_res.loc[~splot_res["Response variable"].str.contains("_ln")]

# For each trait in prim_trait_idx, get the row corresponding to its highest CV r-squared
# Do this by finding the rows in which "Response variable" contains the trait id (e.g. "_X{trait_id}_}")
# Then sort by "CV r-squared" and keep the first row
gbif_best = []
splot_best = []

for trait_id in trait_map.keys():
    gbif_rows = gbif_res.loc[
        gbif_res["Response variable"].str.contains(f"_X{trait_id}_")
    ].sort_values(by="CV r-squared", ascending=False)
    gbif_best_row = gbif_rows.iloc[0]

    splot_rows = splot_res.loc[
        splot_res["Response variable"].str.contains(f"_X{trait_id}_")
    ].sort_values(by="CV r-squared", ascending=False)
    splot_best_row = splot_rows.iloc[0]

    # If the best row for GBIF and sPlot are different (e.g. the best GBIF model is
    # log-transformed and the best sPlot model is not), use the second best row for
    # sPlot. This is because we want to compare the same model for GBIF and sPlot.
    if (
        gbif_best_row["Response variable"].split("GBIF_")[1]
        != splot_best_row["Response variable"].split("sPlot_")[1]
    ):
        splot_best_row = splot_rows.iloc[1]

    # Check again to make sure gbif_best_row and splot_best_row response variables
    # are the same.
    assert (
        gbif_best_row["Response variable"].split("GBIF_")[1]
        == splot_best_row["Response variable"].split("sPlot_")[1]
    ), f"GBIF: {gbif_best_row['Response variable']}\nsPlot: {splot_best_row['Response variable']}"

    gbif_best.append(gbif_best_row)
    splot_best.append(splot_best_row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=str)
    parser.add_argument("--new_imputed", type=str, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--predict-from-row", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--test-run", action="store_true")
    args = parser.parse_args()

    for gbif_row, splot_row in zip(gbif_best, splot_best):
        trained_gbif = TrainedSet.from_results_row(gbif_row)
        trained_splot = TrainedSet.from_results_row(splot_row)

        trait_name = trained_splot.y_name.split("sPlot_")[1]
        gbif_trait = trained_gbif.y_name.split("GBIF_")[1]

        gbif_dir_name = "GBIF_ln" if gbif_trait.endswith("_ln") else "GBIF"
        splot_dir_name = "sPlot_ln" if trait_name.endswith("_ln") else "sPlot"

        new_data = Path(args.new)

        pred_dir = Path(f"./results/predictions/{new_data.stem}/{trait_name}")

        if args.test_run:
            pred_dir = Path(f"./results/predictions/test/{new_data.stem}/{trait_name}")

        gbif_pred_dir = pred_dir / "GBIF"
        splot_pred_dir = pred_dir / "sPlot"
        gbif_pred_dir.mkdir(exist_ok=True, parents=True)
        splot_pred_dir.mkdir(exist_ok=True, parents=True)

        if args.tiled:
            tiles = new_data.glob("*.parq*")

            gbif_pred_dir = gbif_pred_dir / "tiled_5x5_deg"
            splot_pred_dir = splot_pred_dir / "tiled_5x5_deg"
            gbif_pred_dir.mkdir(exist_ok=True, parents=True)
            splot_pred_dir.mkdir(exist_ok=True, parents=True)

            for tile in tiles:
                print(f"Trait: {trait_name}\nTile: {tile.name}")
                # Skip if tile is already fully predicted
                if (
                    Path(gbif_pred_dir, f"{tile.name}_AoA").exists()
                    and Path(splot_pred_dir, f"{tile.name}_AoA").exists()
                ):
                    print(f"Already predicted for {tile.name}")
                    continue

                print(f"Predicting...")
                new_df = dgpd.read_parquet(tile).compute()
                new_df_imp = dgpd.read_parquet(
                    f"{tile.parent}_imputed_fixed/{tile.name}"
                ).compute()
                tile_pred_gbif = Prediction(
                    trained_set=trained_gbif,
                    new_data=new_df,
                    new_data_imputed=new_df_imp,
                )
                tile_pred_splot = Prediction(
                    trained_set=trained_splot,
                    new_data=new_df,
                    new_data_imputed=new_df_imp,
                )
                # Only predict if not already predicted
                if not (gbif_pred_dir / f"{tile.name}_AoA").exists():
                    tile_pred_gbif_df = tile_pred_gbif.df
                    tile_pred_gbif_df.to_parquet(gbif_pred_dir / f"{tile.name}_AoA")
                    print(f"GBIF predicted for {tile.name}")

                if not (splot_pred_dir / f"{tile.name}_AoA").exists():
                    tile_pred_splot_df = tile_pred_splot.df
                    tile_pred_splot_df.to_parquet(splot_pred_dir / f"{tile.name}_AoA")
                    print(f"sPlot predicted for {tile.name}")
        else:
            print(f"Predicting {trait_name}...")

            new_df = gpd.read_parquet(new_data)

            if args.new_imputed is not None:
                new_df_imp = gpd.read_parquet(args.new_imputed)
            else:
                new_df_imp = None

            pred_gbif = Prediction(
                trained_set=trained_gbif,
                new_data=new_df,
                new_data_imputed=new_df_imp,
            )
            pred_splot = Prediction(
                trained_set=trained_splot,
                new_data=new_df,
                new_data_imputed=new_df_imp,
            )

            out_fn = f"{new_data.stem}_predict.parq"
            write_kwargs = {"compression": "zstd", "compression_level": 2}

            # Only predict if not already predicted
            if not (gbif_pred_dir / out_fn).exists() or args.overwrite:
                pred_gbif_df = pred_gbif.df
                pred_gbif_df.to_parquet(gbif_pred_dir / out_fn, **write_kwargs)
                print(f"GBIF predicted for {trait_name}")
            else:
                print(f"Already predicted GBIF for {trait_name}. Skipping...")

            if not (splot_pred_dir / out_fn).exists() or args.overwrite:
                pred_splot_df = pred_splot.df
                pred_splot_df.to_parquet(splot_pred_dir / out_fn, **write_kwargs)
                print(f"sPlot predicted for {trait_name}")
            else:
                print(f"Already predicted sPlot for {trait_name}. Skipping...")
