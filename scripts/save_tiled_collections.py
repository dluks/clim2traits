import argparse
import multiprocessing as mp
from pathlib import Path

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import rioxarray as riox
import xarray as xr

from utils.datasets import CollectionName, Dataset, Unit
from utils.geodata import merge_gdfs, pad_raster

# TODO: fix bug that adds the tiled parent directory name (e.g. "tiled_5x5_deg...") as
# a column to the final GeoDataFrames


def build_grid(tile_size: int = 5) -> list[tuple]:
    """Build grid of coordinates for tiles of size `tile_size` degrees"""
    coords = []
    for tile_lon in np.arange(-180, 180, tile_size):
        for tile_lat in np.arange(-60, 90, tile_size):
            coords.append(
                (tile_lon, tile_lat, tile_lon + tile_size, tile_lat + tile_size)
            )
    return coords


def build_dataframe(ds: xr.Dataset, tile: tuple) -> gpd.GeoDataFrame:
    """Build GeoDataFrame from tile subset of xarray Dataset"""
    # only pad the dataset if its bounds do not already cover the tile
    res = ds.rio.resolution()[0]
    if ds.x.min() > tile[0] + (res / 2) or ds.x.max() < tile[2] - (res / 2):
        print(f"Padding dataset {ds.data_vars}...")
        ds = pad_raster(ds)
    elif ds.y.min() > tile[1] + (res / 2) or ds.y.max() < tile[3] - (res / 2):
        print(f"Padding dataset {ds.data_vars}...")
        ds = pad_raster(ds)

    # Next slice the datasets according to the current tile
    ds = ds.sel(x=slice(tile[0], tile[2]), y=slice(tile[3], tile[1]))
    df = ds.to_dataframe().reset_index().drop(columns=["spatial_ref", "band"])
    ds.close()
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.x, df.y), crs="EPSG:4326")
    df = df.drop(columns=["x", "y"])
    return df


def build_and_write_gdf_tile(
    datasets: list[xr.Dataset], tile: tuple, out_dir: Path, overwrite: bool = False
) -> None:
    """Build and write GeoDataFrame of merged datasets for a given tile"""
    tile_path = out_dir / f"tile_{tile[0]}_{tile[1]}_{tile[2]}_{tile[3]}.parq"

    if not overwrite and (tile_path).exists():
        print(f"Tile {tile} already exists, skipping...")
        return None

    print(f"Processing tile {tile}...")
    tile_gdfs = [build_dataframe(ds, tile) for ds in datasets]
    tile_gdfs = merge_gdfs(tile_gdfs)

    tile_size = tile[2] - tile[0]
    res = datasets[0].rio.resolution()[0]

    # Check that number of rows is equal to the number of pixels in the tile
    if tile_gdfs.shape[0] != (tile_size**2) / res**2:
        raise ValueError(
            f"Number of rows in tile ({tile_gdfs.shape[0]}) does not match number of"
            f" pixels in tile ({(tile_size**2) / res**2})"
        )

    # Set threshold to 50% of non-geometry columns
    thresh = int(tile_gdfs.shape[1] - 1 / 2)

    tile_gdfs = tile_gdfs.dropna(
        thresh=thresh,
        subset=tile_gdfs.columns.difference(["geometry"]),
    )

    # Convert to dask GeoDataFrame for parallel writing
    tile_len = len(tile_gdfs)

    if tile_len == 0:
        print(f"{tile} is empty, skipping...")
        # Append tile tuple to "empty_tiles" file
        with open(out_dir / "empty_tiles.txt", "a", encoding="utf-8") as f:
            f.write(f"{tile}\n")
        return None

    tile_dgdfs = dgpd.from_geopandas(tile_gdfs, npartitions=20)
    del tile_gdfs

    print(f"Writing {tile} with {tile_len} rows...")
    tile_dgdfs.to_parquet(
        tile_path,
        compression="zstd",
        compression_level=1,
    )
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tile_size",
        type=int,
        default=5,
        help="Size of tiles in degrees",
    )
    parser.add_argument(
        "--ds_res",
        type=float,
        default=0.01,
        help="Resolution of datasets in degrees",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing tiles",
    )
    args = parser.parse_args()

    modis = Dataset(
        res=args.ds_res,
        unit=Unit.DEGREE,
        collection_name=CollectionName.MODIS,
    )

    vodca = Dataset(
        res=args.ds_res,
        unit=Unit.DEGREE,
        collection_name=CollectionName.VODCA,
    )

    soil = Dataset(
        res=args.ds_res,
        unit=Unit.DEGREE,
        collection_name=CollectionName.SOIL,
    )

    wc = Dataset(
        res=args.ds_res,
        unit=Unit.DEGREE,
        collection_name=CollectionName.WORLDCLIM,
    )

    coll = [modis, vodca, soil, wc]
    coll_fpaths = [fp for ds in coll for fp in ds.fpaths]

    output_dir = Path(
        f"data/collections/tiled_{args.tile_size}x{args.tile_size}_deg_MOD09GA.061_"
        f"ISRIC_soil_WC_BIO_VODCA_{args.ds_res}_deg_nan-strat=any_thr=0.5"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    tile_coords = build_grid(args.tile_size)

    print("Opening datasets...")
    predictor_datasets = [
        riox.open_rasterio(fp, masked=True).squeeze().to_dataset(name=Path(fp).stem)  # type: ignore
        for fp in coll_fpaths
    ]

    def process_tile(tile):
        """Wrapper function to catch exceptions in the pool"""
        build_and_write_gdf_tile(predictor_datasets, tile, output_dir, args.overwrite)

    with mp.Pool(12) as pool:
        for result in pool.imap_unordered(process_tile, tile_coords):
            print(result)
