from pathlib import Path

import dask_geopandas as dgpd
import pandas as pd

from utils.models import Prediction, TrainedSet

results = pd.read_csv("./results/training_results.csv").copy()

# Key trait ids
# 26 = Seed dry mass
# 11 = SLA
# 4 = SSD
# 18 = Height
# 78 = Leaf Delta 15N
# 169 = Stem conduit density
# 55 = Leaf dry mass
prim_trait_idx = [
    26,
    11,
    4,
    18,
    78,
    169,
    55,
]

# First filter results to only include rows that were not imputed
results = results.loc[results["NaN strategy"] == "all"]

# Filter results to only include rows in which Response variable starts with GBIF
gbif_res = results.loc[results["Response variable"].str.startswith("GBIF")]
splot_res = results.loc[results["Response variable"].str.startswith("sPlot")]

# Filter results to only include rows in which Response variable does not contain "_ln"
gbif_res = gbif_res.loc[~gbif_res["Response variable"].str.contains("_ln")]
splot_res = splot_res.loc[~splot_res["Response variable"].str.contains("_ln")]

# For each trait in prim_trait_idx, get the row corresponding to its highest CV r-squared
# Do this by finding the rows in which "Response variable" contains the trait id (e.g. "_X{trait_id}_}")
# Then sort by "CV r-squared" and keep the first row
gbif_best = []
splot_best = []

for trait_id in prim_trait_idx:
    gbif_best.append(
        gbif_res.loc[gbif_res["Response variable"].str.contains(f"_X{trait_id}_")]
        .sort_values(by="CV r-squared", ascending=False)
        .iloc[0]
    )
    splot_best.append(
        splot_res.loc[splot_res["Response variable"].str.contains(f"_X{trait_id}_")]
        .sort_values(by="CV r-squared", ascending=False)
        .iloc[0]
    )

if __name__ == "__main__":
    for gbif_row, splot_row in zip(gbif_best, splot_best):
        trained_gbif = TrainedSet.from_results_row(gbif_row)
        trained_splot = TrainedSet.from_results_row(splot_row)

        tiles = Path(
            "data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg/"
        ).glob("*.parq*")

        trait_name = trained_splot.y_name.split("sPlot_")[1]

        tiled_pred_dir = Path(
            f"./results/predictions/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg/{trait_name}"
        )
        gbif_pred_dir = tiled_pred_dir / "GBIF" / "tiled_5x5_deg"
        splot_pred_dir = tiled_pred_dir / "sPlot" / "tiled_5x5_deg"
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
                trained_set=trained_gbif, new_data=new_df, new_data_imputed=new_df_imp
            )
            tile_pred_splot = Prediction(
                trained_set=trained_splot, new_data=new_df, new_data_imputed=new_df_imp
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
