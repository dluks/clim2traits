import argparse
import sys
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd

from utils.spatial_stats import impute_missing


def impute_tile(tile_gdf: Path, out_dir: Path, verbose: bool = False):
    """Impute missing values in a single tile"""
    print(f"Imputing {tile_gdf.stem}")
    try:
        gdf = dgpd.read_parquet(tile_gdf).compute()

        # Remove columns containing "tiled" in the name (this is due to a bug in
        # saved_tiled_collections.py)
        gdf = gdf.loc[:, ~gdf.columns.str.contains("tiled")]

        df_imp = impute_missing(gdf.drop(columns=["geometry"]), verbose=verbose)
        gdf = gpd.GeoDataFrame(
            df_imp.reset_index(drop=True),
            geometry=gdf.reset_index(drop=True).geometry,
            crs=gdf.crs,
        )
        dgdf_imp = dgpd.from_geopandas(gdf, npartitions=20)
        dgdf_imp.to_parquet(
            Path(out_dir, tile_gdf.name), compression="zstd", compression_level=2
        )
        print(f"Imputed {tile_gdf.stem}")
    except Exception as e:
        print(f"Failed to impute {tile_gdf.stem}")
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "collection",
        type=str,
        help="Path to tiled collection",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Print verbose output",
    )

    args = parser.parse_args()

    collection_path = Path(args.collection)

    # Create imputed collection dir
    imputed_dir = collection_path.parent / f"{collection_path.name}_imputed"
    imputed_dir.mkdir(exist_ok=True, parents=True)

    tiles = collection_path.glob("*.parq*")

    for tile in tiles:
        impute_tile(tile, imputed_dir, args.v)
    sys.exit(0)
