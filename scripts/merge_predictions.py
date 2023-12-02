import argparse
import logging
import multiprocessing as mp
from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dgpd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def merge_predict_tiles(
    trait_dir: Path, trait_set: str = "GBIF", overwrite: bool = False
):
    """Merge predictions for a given trait set.

    Args:
        trait_dir (Path): Path to the trait directory.
        trait_set (str): Name of the trait set. Defaults to "GBIF". Must be one of
            "GBIF" or "sPlot".
    """
    if not trait_dir.is_dir():
        raise ValueError(f"Invalid trait directory: {trait_dir}")

    if not overwrite:
        if Path(trait_dir, f"{trait_set}/merged_predictions.parq").exists():
            logging.info(
                "Predictions already merged for %s %s", trait_dir.name, trait_set
            )
            return

    logging.info("Merging predictions for %s %s", trait_dir.name, trait_set)
    tiles = list(Path(trait_dir, f"{trait_set}/tiled_5x5_deg").glob("*.parq"))
    gdfs = [dgpd.read_parquet(tile, npartitions=20) for tile in tiles]
    gdfs = dd.concat(gdfs)
    gdfs.to_parquet(  # type: ignore
        Path(trait_dir, f"{trait_set}/merged_predictions.parq"),
        compression="zstd",
        compression_level=2,
    )
    logging.info("Predictions merged for %s %s", trait_dir.name, trait_set)


def main():
    """Merge tiled predictions for all traits."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions", type=str, help="Path to the predictions directory."
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite merged predictions if they already exist.",
    )
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes.")
    args = parser.parse_args()

    if args.num_procs == -1:
        args.num_procs = mp.cpu_count()

    trait_dirs = Path(args.predictions).glob("*")

    with mp.Pool(args.num_procs) as pool:
        for td in trait_dirs:
            pool.apply_async(merge_predict_tiles, args=(td, "GBIF", args.overwrite))
            pool.apply_async(merge_predict_tiles, args=(td, "sPlot", args.overwrite))
        pool.close()
        pool.join()


if __name__ == "__main__":
    main()
