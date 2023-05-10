import glob
import os

from osgeo import gdal, gdalconst
from tqdm import tqdm

files = sorted(glob.glob("./MODIS/*.tif"))

kwargs = {"format": "GTiff", "outputType": gdalconst.GDT_Int16}

j = 0
for i, fn in tqdm(enumerate(files), total=len(files) / 2):
    if i % 2 == 0:
        pair = [files[i], files[i + 1]]
        name = os.path.splitext(os.path.basename(files[i]))[0].split("-0000000000")[0]
        # print(name)
        out_fn = os.path.join("./MODIS/merged/", f"{name}.tif")
        ds = gdal.Warp(out_fn, pair, **kwargs)
