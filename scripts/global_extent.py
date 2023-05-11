#!/usr/bin/env python3

from osgeo import gdal
from tqdm import tqdm

from PreprocessingConfig import Config

config = Config()

kwargs = {
    "outputBounds": (-180, -90, 180, 90),
    "creationOptions": [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
    ],
}

fns = config.MODIS_fns(res="1km")

for fn in tqdm(fns):
    # name = os.path.splitext(os.path.basename(fn))[0]
    # dirname = os.path.dirname(fn)
    # out_fn = os.path.join(dirname, f"{name}_full.tif")
    ds = gdal.Warp(fn, fn, **kwargs)
