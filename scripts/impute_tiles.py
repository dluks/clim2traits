import argparse
import multiprocessing as mp
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd

from utils.spatial_stats import impute_missing


def impute_tile(tile, out_dir):
    """Impute missing values in a single tile"""
    print(f"Imputing {tile.stem}")
    gdf = dgpd.read_parquet(tile).compute()
    df_imp = impute_missing(gdf.drop(columns=["geometry"]), verbose=False)
    gdf = gpd.GeoDataFrame(df_imp, geometry=gdf.geometry, crs=gdf.crs)
    dgdf_imp = dgpd.from_geopandas(gdf, npartitions=20)
    dgdf_imp.to_parquet(
        Path(out_dir, tile.name), compression="zstd", compression_level=2
    )
    print(f"Imputed {tile.stem}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collection",
        type=str,
        help="Path to tiled collection",
    )
    parser.add_argument("--mp", action="store_true")

    args = parser.parse_args()

    collection_path = Path(args.collection)

    # Create imputed collection dir
    imputed_dir = collection_path.parent / f"{collection_path.name}_imputed"
    imputed_dir.mkdir(exist_ok=True, parents=True)

    tiles = collection_path.glob("*.parq*")

    if not args.mp:
        for tile in tiles:
            impute_tile(tile, imputed_dir)
        exit()
    else:
        print("Running in multiprocessing mode")
        # Create a pool of worker processes
        num_processes = mp.cpu_count()
        with mp.Pool(num_processes) as pool:
            # Apply the impute_tile function to each tile in parallel
            results = [
                pool.apply_async(impute_tile, args=(tile, imputed_dir))
                for tile in tiles
            ]

            # Wait for all worker processes to finish
            for result in results:
                result.wait()
