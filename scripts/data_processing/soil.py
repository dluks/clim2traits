#!/usr/bin/env python3
import argparse
import gc

import rioxarray as riox

from utils.dataset_tools import Unit
from utils.datasets import CollectionName, Dataset, resample_dataset

soil = Dataset(res=1, unit=Unit.KILOMETER, collection_name=CollectionName.SOIL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--resample_05", action="store_true")
    parser.add_argument("--resample_001", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.standardize:
        # Standardize the files
        for fp in soil.fpaths:
            out_fpath = fp
            # fp = Path(fp)
            # parent_dir = fp.parent
            # out_dir = parent_dir / "standardized"
            # out_dir.mkdir(exist_ok=True)
            # out_fpath = out_dir / f"{fp.stem}.tif"

            if args.dry_run:
                print(f"Would write to {out_fpath}")
                continue

            ds = riox.open_rasterio(fp, chunks={"x": 360, "y": 360})
            ds.rio.write_crs(ds.rio.crs, inplace=True)
            ds = ds.astype("int16")
            ds = ds.where(ds != ds.rio.nodata)
            ds.rio.write_nodata(ds.rio.nodata, encoded=True, inplace=True)
            ds.rio.to_raster(
                out_fpath,
                dtype="int16",
                compress="zstd",
                tiled=True,
                blockxsize=256,
                blockysize=256,
                predictor=2,
                num_threads=20,
                windowed=True,
                compute=False,
            )
            print(f"Wrote to {out_fpath}")

            ds.close()
            del ds
            gc.collect()

    if args.resample_05:
        resample_dataset(
            soil,
            0.5,
            Unit.DEGREE,
            resample_alg=5,
            match_raster=True,
            dry_run=args.dry_run,
        )

    if args.resample_001:
        resample_dataset(soil, 0.01, Unit.DEGREE, resample_alg=5, dry_run=args.dry_run)
