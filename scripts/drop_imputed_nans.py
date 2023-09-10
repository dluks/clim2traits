import multiprocessing as mp
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd


def drop_nans(imp_tile_path, src_tile_path):
    src_tile = dgpd.read_parquet(src_tile_path).compute().reset_index(drop=True)
    imp_tile = dgpd.read_parquet(imp_tile_path).compute().reset_index(drop=True)

    imp_tile = imp_tile.drop(columns=["geometry"])
    imp_tile = imp_tile.dropna(axis=0, how="all")

    imp_tile = gpd.GeoDataFrame(imp_tile, geometry=src_tile.geometry, crs=src_tile.crs)

    if imp_tile.shape != src_tile.shape:
        raise ValueError(f"Shapes do not match: {imp_tile.shape} != {src_tile.shape}")

    if imp_tile.dropna(axis=1, how="all").isnull().values.any():
        raise ValueError(f"NaN rows still present in imputed tile {imp_tile_path.name}")

    imp_tile = dgpd.from_geopandas(imp_tile, npartitions=20)

    out_dir = Path(f"{imp_tile_path.parent}_fixed")
    out_dir.mkdir(exist_ok=True, parents=True)

    imp_tile.to_parquet(
        out_dir / imp_tile_path.name, compression="zstd", compression_level=2
    )


src_dir = Path(
    "data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg/"
)
imputed_dir = Path(
    "data/collections/tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_imputed/"
)

imp_tiles = imputed_dir.glob("*.parq*")

if __name__ == "__main__":
    with mp.Pool(mp.cpu_count()) as pool:
        results = [
            pool.apply_async(drop_nans, args=(tile, src_dir / tile.name))
            for tile in imp_tiles
        ]

        for result in results:
            result.get()
