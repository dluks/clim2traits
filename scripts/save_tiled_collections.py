import multiprocessing as mp
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import rioxarray as riox
import xarray as xr

from utils.datasets import CollectionName, Dataset, Unit
from utils.geodata import merge_gdfs

TILE_SIZE = 5
DS_RES = 0.01


def generate_coords(tile_size: int = 5) -> list[tuple]:
    coords = []
    for tile_lon in np.arange(-180, 180, tile_size):
        for tile_lat in np.arange(-60, 90, tile_size):
            coords.append(
                (tile_lon, tile_lat, tile_lon + tile_size, tile_lat + tile_size)
            )
    return coords


def build_dataframe(ds: xr.Dataset, tile: tuple) -> gpd.GeoDataFrame:
    # First slice the datasets according to the current tile
    ds = ds.sel(x=slice(tile[0], tile[2]), y=slice(tile[3], tile[1]))
    df = ds.to_dataframe().reset_index().drop(columns=["spatial_ref", "band"])
    ds.close()
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")
    df = df.drop(columns=["x", "y"])
    return df


def build_and_write_gdf_tile(datasets: list[xr.Dataset], tile: tuple) -> None:
    print(f"Processing tile {tile}...")
    tile_gdfs = [build_dataframe(ds, tile) for ds in datasets]

    tile_gdfs = merge_gdfs(tile_gdfs)

    # Check that number of rows is equal to the number of pixels in the tile
    if tile_gdfs.shape[0] != (TILE_SIZE**2) / DS_RES**2:
        raise ValueError(
            f"Number of rows in tile ({tile_gdfs.shape[0]}) does not match number of pixels in tile ({(TILE_SIZE**2) / DS_RES**2})"
        )

    # Convert to dask GeoDataFrame for parallel writing
    tile_gdfs = dgpd.from_geopandas(tile_gdfs, npartitions=20)
    tile_gdfs = tile_gdfs.dropna(
        how="all", subset=tile_gdfs.columns.difference(["geometry"])
    )
    tile_len = len(tile_gdfs)

    if tile_len == 0:
        print(f"{tile} is empty, skipping...")
        return None
    else:
        print(f"Writing {tile} with {tile_len} rows...")
        tile_gdfs.to_parquet(
            f"{output_dir}/tile_{tile[0]}_{tile[1]}_{tile[2]}_{tile[3]}.parq",
            compression="zstd",
            compression_level=1,
        )
        return None


modis = Dataset(
    res=DS_RES,
    unit=Unit.DEGREE,
    collection_name=CollectionName.MODIS,
)

vodca = Dataset(
    res=DS_RES,
    unit=Unit.DEGREE,
    collection_name=CollectionName.VODCA,
)

soil = Dataset(
    res=DS_RES,
    unit=Unit.DEGREE,
    collection_name=CollectionName.SOIL,
)

wc = Dataset(
    res=DS_RES,
    unit=Unit.DEGREE,
    collection_name=CollectionName.WORLDCLIM,
)

coll = [modis, vodca, soil, wc]
coll_fpaths = [fp for ds in coll for fp in ds.fpaths]

output_dir = Path(
    f"data/collections/tiled_{TILE_SIZE}x{TILE_SIZE}_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_{DS_RES}_deg"
)
output_dir.mkdir(exist_ok=True, parents=True)


if __name__ == "__main__":
    tile_coords = generate_coords(TILE_SIZE)
    print("Opening datasets...")
    datasets = [
        riox.open_rasterio(fp, masked=True).squeeze().to_dataset(name=Path(fp).stem)
        for fp in coll_fpaths
    ]

    def process_tile(tile):
        build_and_write_gdf_tile(datasets, tile)

    with mp.Pool() as pool:
        results = [pool.apply_async(process_tile, args=(tile,)) for tile in tile_coords]
        [result.get() for result in results]
        pool.close()
        pool.join()
