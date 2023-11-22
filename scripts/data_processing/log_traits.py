import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import rioxarray as riox
import xarray as xr

logging.basicConfig(level=logging.INFO)


def log_transform_trait(
    fp: Union[str, os.PathLike], band: Optional[int]
) -> xr.DataArray:
    """Log-transform a trait DataArray."""
    ds = riox.open_rasterio(fp, masked=True)

    if isinstance(ds, list):
        raise ValueError("Input file is a list. Please provide a single file.")

    if band is not None:
        ds = ds.sel(band=band).assign_attrs(
            {"long_name": ds.attrs["long_name"][band - 1]}
        )

    # Log-transform
    ds_log = ds.copy()
    ds_log.data = np.log(ds.data)

    return xr.DataArray(ds_log)


def check_for_inf(fp: Union[str, os.PathLike]) -> bool:
    """Check if a file contains inf values."""
    ds = riox.open_rasterio(fp, masked=True)

    if isinstance(ds, list):
        raise ValueError("Input file is a list. Please provide a single file.")

    return np.any(np.isinf(ds.data))


def main():
    """Log-transform trait maps."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    pft_dirs = Path("GBIF_trait_maps/global_maps").glob("*")
    pft_dirs = [d for d in pft_dirs if d.is_dir()]

    all_files = [pft_dir.glob("*/*.grd") for pft_dir in pft_dirs]
    all_files = [f for l in all_files for f in l]  # Flatten list

    write_args = {
        "compress": "zstd",
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "predictor": 2,
        "num_threads": 20,
        "windowed": True,
        "compute": False,
    }

    for f in all_files:
        fn = f"{f.stem}_mean_ln.tif"
        out_dir = Path(
            "data/GBIF_trait_maps_ln",
            f.relative_to("GBIF_trait_maps/global_maps").parent,
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / fn

        if args.check:
            # Check for inf values in data
            if check_for_inf(out):
                logging.warning("Found inf values in %s", out)
            else:
                logging.info("No inf values found in %s", out)
            continue

        ds_lg = log_transform_trait(f, band=2)
        # Drop "band" coord from DataArray
        ds_lg = ds_lg.squeeze().drop_vars("band")

        ds_lg.rio.to_raster(out, **write_args)

        if args.verbose:
            logging.info("Exported %s", out)


if __name__ == "__main__":
    main()
