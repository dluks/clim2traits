import argparse
import logging
from multiprocessing import Pool
from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dgpd

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def merge_predict_tiles(trait_dir: Path, trait_set: str = "GBIF"):
    """Merge predictions for a given trait set.

    Args:
        trait_dir (Path): Path to the trait directory.
        trait_set (str): Name of the trait set. Defaults to "GBIF". Must be one of "GBIF" or "sPlot".
    """
    logging.info("Merging predictions for %s %s", trait_dir.name, trait_set)
    tiles = list(Path(trait_dir, f"{trait_set}/tiled_5x5_deg").glob("*.parq"))
    gdfs = [dgpd.read_parquet(tile, npartitions=20) for tile in tiles]
    gdfs = dd.concat(gdfs)
    gdfs.to_parquet(
        Path(trait_dir, f"{trait_set}/merged_predictions.parq"),
        compression="zstd",
        compression_level=2,
    )
    logging.info("Predictions merged for %s %s", trait_dir.name, trait_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "predictions", type=str, help="Path to the predictions directory."
    )
    args = parser.parse_args()

    trait_dirs = Path(args.predictions).glob("*")

    with Pool() as pool:
        for td in trait_dirs:
            pool.apply_async(merge_predict_tiles, args=(td, "GBIF"))
            pool.apply_async(merge_predict_tiles, args=(td, "sPlot"))
        pool.close()
        pool.join()
