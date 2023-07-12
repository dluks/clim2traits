#!/usr/bin/env python3
import glob
import os

from osgeo import gdal, gdalconst
from tqdm import tqdm

dd = "/DATA/lusk/thesis/traits/data/modis/gee/multiband_monthly_averages"
files = sorted(glob.glob(os.path.join(dd, "*.tif")))

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
def gdal_warp(out_fn, pair, kwargs):
    ds = gdal.Warp(out_fn, pair, **kwargs)
    return ds


j = 0
for i, fn in tqdm(enumerate(files), total=len(files) // 2):
    if i % 2 == 0:
        pair = [files[i], files[i + 1]]
        name = os.path.splitext(os.path.basename(files[i]))[0].split("-0000000000")[0]
        out_fn = os.path.join(dd, "merged", f"{name}.tif")
        ds = gdal_warp(out_fn, pair, kwargs)
        del ds
