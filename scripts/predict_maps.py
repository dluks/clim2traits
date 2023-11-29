import argparse
import json
import multiprocessing as mp
from dataclasses import dataclass
from functools import cached_property
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
    results = pd.read_csv("./results/training_results.csv.gz").copy()

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


@dataclass
class PredictionDirs:
    """Class for storing prediction output directories"""

    gbif: Path
    splot: Path


@dataclass
class TrainedTraitInfo:
    """Class for storing trait information"""

    name: str
    gbif_trained_set: TrainedSet
    splot_trained_set: TrainedSet


@dataclass
class Tile:
    """Class for storing tile information"""

    path: Path
    calc_aoa: bool = False

    @cached_property
    def df(self) -> gpd.GeoDataFrame:
        """Return the tile as a GeoDataFrame"""
        df = dgpd.read_parquet(self.path).compute().reset_index(drop=True)
        # Remove columns containing "tiled" in the name (this is due to a bug in
        # saved_tiled_collections.py)
        df = df.loc[:, ~df.columns.str.contains("tiled")]
        return df

    @cached_property
    def df_imputed(self) -> gpd.GeoDataFrame:
        """Return the imputed tile as a GeoDataFrame"""
        imputed_path = Path(f"{self.path.parent}_imputed", self.path.name)
        return dgpd.read_parquet(imputed_path).compute().reset_index(drop=True)


def predict_tile(
    tile: Tile,
    prediction_dirs: PredictionDirs,
    trained_trait_info: TrainedTraitInfo,
    overwrite: bool = False,
) -> None:
    """Predict a trait for a single tile"""
    print(f"Trait: {trained_trait_info.name}\nTile: {tile.path.name}\n")
    # Skip if tile is already fully predicted
    if (
        Path(prediction_dirs.gbif, tile.path.name).exists()
        and Path(prediction_dirs.splot, tile.path.name).exists()
        and not overwrite
    ):
        print(f"Already predicted for {tile.path.name}")
        return

    print(f"Predicting {tile.path.stem}...")

    tile_pred_gbif = Prediction(
        trained_set=trained_trait_info.gbif_trained_set,
        new_data=tile.df,
        new_data_imputed=tile.df_imputed,
        calc_aoa=tile.calc_aoa,
    )

    tile_pred_splot = Prediction(
        trained_set=trained_trait_info.splot_trained_set,
        new_data=tile.df,
        new_data_imputed=tile.df_imputed,
        calc_aoa=tile.calc_aoa,
    )

    # Only predict if not already predicted
    if not (prediction_dirs.gbif / tile.path.name).exists() or overwrite:
        tile_pred_gbif_df = tile_pred_gbif.df
        tile_pred_gbif_df.to_parquet(
            prediction_dirs.gbif / tile.path.name,
            compression="zstd",
            compression_level=2,
        )
        print(f"GBIF predicted for {tile.path.stem}")

    if not (prediction_dirs.splot / tile.path.name).exists() or overwrite:
        tile_pred_splot_df = tile_pred_splot.df
        tile_pred_splot_df.to_parquet(
            prediction_dirs.splot / tile.path.name,
            compression="zstd",
            compression_level=2,
        )
        print(f"sPlot predicted for {tile.path.stem}")


def main():
    """Predict traits for new data"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--new", type=str)
    parser.add_argument("--new-imputed", type=str, default=None)
    parser.add_argument(
        "-r", "--model-res", type=float, default=0.5, help="Model resolution"
    )
    parser.add_argument(
        "-p", "--pft", type=str, default="Shrub_Tree_Grass", help="Model PFT"
    )
    parser.add_argument("--run-ids", nargs="+", type=str, default=None)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("-o", "--overwrite", action="store_true")
    parser.add_argument("--test-run", action="store_true")
    parser.add_argument("--trait-ids", nargs="+", type=int, default=None)
    parser.add_argument(
        "--aoa", action="store_true", help="Calculate AOA for predictions"
    )
    parser.add_argument(
        "--num-procs",
        type=int,
        default=1,
        help="Number of processes (> 1 will use multiprocessing; -1 will use all"
        "available cores)",
    )

    args = parser.parse_args()

    if args.num_procs == -1:
        args.num_procs = mp.cpu_count()

    gbif_best, splot_best = get_best_models(args.model_res, args.pft, args.run_ids)

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

        trait_name = trained_gbif.y_name.split("GBIF_")[1]

        new_data = Path(args.new)

        model_dir = Path(
            "./results/predictions", f"{num_to_str(args.model_res)}deg_models"
        )

        pred_dir = Path(
            model_dir,
            new_data.stem,
            args.pft,
            trait_name,
        )

        if args.tiled:
            pred_dir = Path(model_dir, new_data.name, args.pft, trait_name)

        if args.test_run:
            pred_dir = Path(
                model_dir.parent,
                "test",
                model_dir.name,
                new_data.name,
                args.pft,
                trait_name,
            )

        gbif_pred_dir = pred_dir / "GBIF"
        splot_pred_dir = pred_dir / "sPlot"
        gbif_pred_dir.mkdir(exist_ok=True, parents=True)
        splot_pred_dir.mkdir(exist_ok=True, parents=True)

        if args.tiled:
            tile_paths = new_data.glob("*.parq*")

            gbif_pred_dir = gbif_pred_dir / "tiled_5x5_deg"
            splot_pred_dir = splot_pred_dir / "tiled_5x5_deg"
            gbif_pred_dir.mkdir(exist_ok=True, parents=True)
            splot_pred_dir.mkdir(exist_ok=True, parents=True)

            prediction_dirs = PredictionDirs(gbif=gbif_pred_dir, splot=splot_pred_dir)

            trained_trait_info = TrainedTraitInfo(
                name=trait_name,
                gbif_trained_set=trained_gbif,
                splot_trained_set=trained_splot,
            )

            if not args.overwrite:
                # Build list of tiles that have not already been predicted
                tile_paths = [
                    tile_path
                    for tile_path in tile_paths
                    if not (
                        Path(prediction_dirs.gbif, tile_path.name).exists()
                        and Path(prediction_dirs.splot, tile_path.name).exists()
                    )
                ]

            if len(list(tile_paths)) == 0:
                print(f"Already predicted for {trait_name}. Skipping...")
                continue

            if args.num_procs > 1:
                with mp.Pool(args.num_procs) as pool:
                    pool.starmap(
                        predict_tile,
                        [
                            (
                                Tile(tile_path, args.aoa),
                                prediction_dirs,
                                trained_trait_info,
                                args.overwrite,
                            )
                            for tile_path in tile_paths
                        ],
                    )
            elif args.num_procs == 1:
                for tile_path in tile_paths:
                    predict_tile(
                        Tile(tile_path, args.aoa),
                        prediction_dirs,
                        trained_trait_info,
                        args.overwrite,
                    )
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
                calc_aoa=args.aoa,
            )
            pred_splot = Prediction(
                trained_set=trained_splot,
                new_data=new_df,
                new_data_imputed=new_df_imp,
                calc_aoa=args.aoa,
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


if __name__ == "__main__":
    main()
