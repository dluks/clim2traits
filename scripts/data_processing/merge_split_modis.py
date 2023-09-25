#!/usr/bin/env python3
import argparse
from pathlib import Path

import rioxarray as riox
from osgeo import gdal, gdalconst
from tqdm import tqdm

from utils.dataset_tools import Unit
from utils.datasets import CollectionName, Dataset, resample_dataset


def merge_modis(split_tifs: list[Path], out_dir: Path) -> None:
    kwargs = {
        "format": "GTiff",
        "outputType": gdalconst.GDT_Int16,
        "srcNodata": -32768,
        "dstNodata": -32768,
        "multithread": True,
        "creationOptions": [
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
            "BIGTIFF=YES",
        ],
    }

    # @catch_gdal
    # def gdal_warp(out_fn, pair, kwargs):
    #     ds = gdal.Warp(out_fn, pair, **kwargs)
    #     return ds

    for i, _ in tqdm(enumerate(split_tifs), total=len(split_tifs) // 2):
        if i % 2 == 0:
            pair = [str(split_tifs[i]), str(split_tifs[i + 1])]
            name = Path(split_tifs[i]).stem.split("-0000000000")[0]
            out_fn = out_dir / f"{name}.tif"
            ds = gdal.Warp(str(out_fn), pair, **kwargs)
            del ds

            print(f"Exported {out_fn.name}")


def modis_multiband_split(raster: Path, out_dir: Path) -> None:
    month = raster.stem.split("_")[-1]
    ds = riox.open_rasterio(raster)

    for i, arr in enumerate(ds):
        band_num = f"{i + 1:02d}"
        arr.rio.write_crs("EPSG:4326", inplace=True)
        arr = arr.astype("float32")
        arr = arr.where(arr != arr.rio.nodata)
        arr.rio.write_nodata(arr.rio.nodata, encoded=True, inplace=True)
        arr.attrs[
            "long_name"
        ] = f"MODIS 2000-2020 monthly mean surface reflectance band {band_num} - month {month}"

        fname = f"2000-2020_sur_refl_multiyear_mean_{month}_band{band_num}.tif"

        arr.rio.to_raster(
            out_dir / fname,
            dtype="int16",
            compress="zstd",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            predictor=2,
            num_threads=18,
        )

        print(f"Exported {fname}")


if __name__ == "__main__":
    # Accept arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--merge", action="store_true")
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--resample_05", action="store_true")
    parser.add_argument("--resample_001", action="store_true")

    args = parser.parse_args()

    split_tifs_dir = Path(
        "/DATA/lusk/thesis/traits/data/modis/gee/2000-2020/multiband_monthly_averages"
    )
    split_tifs = sorted(list(split_tifs_dir.glob("*.tif")))
    merged_out_dir = Path(split_tifs[0].parent) / "merged"
    merged_out_dir.mkdir(exist_ok=True, parents=True)

    merged_tifs = sorted(list(merged_out_dir.glob("*.tif")))
    out_dir = Path("/DATA/lusk/thesis/traits/data/modis/2000-2020/1_km")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Merge tifs
    if args.merge:
        print("\nMerging tifs...")
        merge_modis(split_tifs, merged_out_dir)

    # 2. Split merged tifs
    if args.split:
        print("\nSplitting merged tifs...")

        for tif in merged_tifs:
            modis_multiband_split(tif, out_dir)

    modis_1km = Dataset(
        res=1,
        unit=Unit.KILOMETER,
        collection_name=CollectionName.MODIS,
    )

    # 3. Resample to 0.5 degree
    if args.resample_05:
        print("\nResampling to 0.5 degree...")
        resample_dataset(modis_1km, 0.5, Unit.DEGREE, resample_alg=5, match_raster=True)

    # 4. Resample to 0.01 degree
    if args.resample_001:
        print("\nResampling to 0.01 degree...")
        resample_dataset(modis_1km, 0.01, Unit.DEGREE)
