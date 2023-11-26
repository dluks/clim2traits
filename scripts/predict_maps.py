import argparse
import json
from pathlib import Path
from typing import Optional, Union

import dask_geopandas as dgpd
import geopandas as gpd
import pandas as pd

from utils.geodata import num_to_str
from utils.models import Prediction, TrainedSet


def get_best_models(
    model_resolution: Union[int, float],
    model_pft: str,
    run_ids: Optional[list[str]] = None,
) -> tuple[list[pd.Series], list[pd.Series]]:
    """Get the best models for each trait from the training results"""
    results = pd.read_csv("./results/training_results.csv").copy()

    with open("trait_mapping.json", "r", encoding="utf-8") as f:
        trait_map = json.load(f)

    # First filter results to only include rows that were not imputed. Only get rows where
    # "NaN strategy" does not contain "imputed"
    results = results.loc[~results["NaN strategy"].str.contains("imputed")]

    if run_ids is not None:
        results = results.loc[results["Run ID"].isin(run_ids)]

    results = results.loc[results["Resolution"] == f"{model_resolution:g}_deg"]
    results = results.loc[results["PFT"] == model_pft]

    assert len(results) > 0, "No models found for specified resolution and PFT"

    gbif_res = results.loc[results["Response variable"].str.startswith("GBIF")]
    splot_res = results.loc[results["Response variable"].str.startswith("sPlot")]

    # For each trait in prim_trait_idx, get the row corresponding to its highest CV r-squared
    # Do this by finding the rows in which "Response variable" contains the trait id
    # (e.g. "_X{trait_id}_}"). Then sort by "CV r-squared" and keep the first row
    gbif_best_rows = []
    splot_best_rows = []

    for trait_id in trait_map.keys():
        gbif_best_row = (
            gbif_res.loc[gbif_res["Response variable"].str.contains(f"_X{trait_id}_")]
            .sort_values(by="CV r-squared", ascending=False)
            .iloc[0]
        )

        splot_best_row = (
            splot_res.loc[
                (
                    (splot_res["Response variable"].str.contains(f"_X{trait_id}_"))
                    & (splot_res.Transform == gbif_best_row.Transform)
                )
            ]
            .sort_values(by="CV r-squared", ascending=False)
            .iloc[0]
        )

        # If the best row for GBIF and sPlot are different (e.g. the best GBIF model is
        # log-transformed and the best sPlot model is not), use the second best row (i.e.
        # the oppositely-transformed model) for sPlot. This is because we want to compare
        # the same model for GBIF and sPlot.
        gbif_trait_name = gbif_best_row["Response variable"].split("GBIF_")[1]
        splot_trait_name = splot_best_row["Response variable"].split("sPlot_")[1]

        assert gbif_trait_name == splot_trait_name, (
            "Mismatched trait names:\n"
            f"GBIF: {gbif_best_row['Response variable']}\n"
            f"sPlot: {splot_best_row['Response variable']}"
        )

        gbif_best_rows.append(gbif_best_row)
        splot_best_rows.append(splot_best_row)

    return (gbif_best_rows, splot_best_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=str)
    parser.add_argument("--new-imputed", type=str, default=None)
    parser.add_argument("-r", "--res", type=float, default=0.5, help="Model resolution")
    parser.add_argument(
        "-p", "--pft", type=str, default="Shrub_Tree_Grass", help="Model PFT"
    )
    parser.add_argument("--run-ids", nargs="+", type=str, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--trait-ids", nargs="+", type=int, default=None)
    args = parser.parse_args()

    gbif_best, splot_best = get_best_models(args.res, args.pft, args.run_ids)

    for gbif_row, splot_row in zip(gbif_best, splot_best):
        # If trait_ids is not None, only predict for the specified trait ids
        if args.trait_ids is not None:
            gbif_trait_id = int(
                gbif_row["Response variable"].split("_X")[1].split("_")[0]
            )
            splot_trait_id = int(
                splot_row["Response variable"].split("_X")[1].split("_")[0]
            )

            if (
                gbif_trait_id not in args.trait_ids
                and splot_trait_id not in args.trait_ids
            ):
                continue

        trained_gbif = TrainedSet.from_results_row(gbif_row)
        trained_splot = TrainedSet.from_results_row(splot_row)

        splot_trait = trained_splot.y_name.split("sPlot_")[1]
        gbif_trait = trained_gbif.y_name.split("GBIF_")[1]
        trait_name = gbif_trait

        gbif_dir_name = "GBIF_ln" if gbif_trait.endswith("_ln") else "GBIF"
        splot_dir_name = "sPlot_ln" if trait_name.endswith("_ln") else "sPlot"

        new_data = Path(args.new)

        pred_dir = Path(
            "./results/predictions",
            f"{num_to_str(args.res)}deg_models",
            new_data.stem,
            args.pft,
            trait_name,
        )

        # TODO: Add model resolution and PFT to tiled directory pred_dir
        if args.tiled:
            pred_dir = Path(f"./results/predictions/{new_data.name}/{trait_name}")

        if args.test_run:
            pred_dir = Path(
                "./results/predictions/test/",
                f"{num_to_str(args.res)}deg_models",
                new_data.name,
                args.pft,
                trait_name,
            )

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
                print(f"Trait: {trait_name}\nTile: {tile.name}\n")
                # Skip if tile is already fully predicted
                if (
                    Path(gbif_pred_dir, tile.name).exists()
                    and Path(splot_pred_dir, tile.name).exists()
                    and not args.overwrite
                ):
                    print(f"Already predicted for {tile.name}")
                    continue

                print(f"Predicting {tile.stem}...")
                new_df = dgpd.read_parquet(tile).compute().reset_index(drop=True)

                # Remove columns containing "tiled" in the name (this is due to a bug in
                # saved_tiled_collections.py)
                new_df = new_df.loc[:, ~new_df.columns.str.contains("tiled")]

                new_df_imp = (
                    dgpd.read_parquet(f"{tile.parent}_imputed/{tile.name}")
                    .compute()
                    .reset_index(drop=True)
                )

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
                if not (gbif_pred_dir / tile.name).exists() or args.overwrite:
                    tile_pred_gbif_df = tile_pred_gbif.df
                    tile_pred_gbif_df.to_parquet(
                        gbif_pred_dir / tile.name,
                        compression="zstd",
                        compression_level=2,
                    )
                    print(f"GBIF predicted for {tile.stem}")

                if not (splot_pred_dir / tile.name).exists() or args.overwrite:
                    tile_pred_splot_df = tile_pred_splot.df
                    tile_pred_splot_df.to_parquet(
                        splot_pred_dir / tile.name,
                        compression="zstd",
                        compression_level=2,
                    )
                    print(f"sPlot predicted for {tile.stem}")
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
            gbif_out_path = gbif_pred_dir / out_fn
            splot_out_path = splot_pred_dir / out_fn

            # Only predict if not already predicted
            if not gbif_out_path.exists() or args.overwrite:
                pred_gbif_df = pred_gbif.df
                pred_gbif_df.to_parquet(gbif_out_path, **write_kwargs)
                print(f"GBIF predicted for {trait_name}")
            else:
                print(f"Already predicted GBIF for {trait_name}. Skipping...")

            if not splot_out_path.exists() or args.overwrite:
                pred_splot_df = pred_splot.df
                pred_splot_df.to_parquet(splot_out_path, **write_kwargs)
                print(f"sPlot predicted for {trait_name}")
            else:
                print(f"Already predicted sPlot for {trait_name}. Skipping...")
