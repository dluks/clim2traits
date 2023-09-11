from multiprocessing import Pool
from pathlib import Path

import dask.dataframe as dd
import dask_geopandas as dgpd


def merge_predict_tiles(trait_dir, trait_set):
    tiles = list(Path(trait_dir, f"{trait_set}/tiled_5x5_deg").glob("*_AoA.parq"))
    gdfs = [dgpd.read_parquet(tile, npartitions=20) for tile in tiles]
    gdfs = dd.concat(gdfs)
    gdfs.to_parquet(
        Path(trait_dir, f"{trait_set}/tiled_5x5_deg_merged.parq"),
        compression="zstd",
        compression_level=2,
    )


if __name__ == "__main__":
    trait_dirs = Path(
        "results/predictions/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg/"
    ).glob("*")

    with Pool() as pool:
        for trait_dir in trait_dirs:
            pool.apply_async(merge_predict_tiles, args=(trait_dir, "GBIF"))
            pool.apply_async(merge_predict_tiles, args=(trait_dir, "sPlot"))
        pool.close()
        pool.join()
