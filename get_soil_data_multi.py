#!/usr/bin/env python3
import logging
import os
import time
from functools import wraps
from multiprocessing import Pool

from osgeo import gdal

logging.basicConfig(
    level=logging.INFO,
    filename="get_soil_data.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


def retry(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        max_tries = 15
        for i in range(max_tries):
            fn = os.path.basename(args[1])
            try:
                print(f"Processing {os.path.basename(fn)}... Attempt: {i+1}")
                return f(*args, **kwargs)
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
                logging.warning(message)
                time.sleep(60)
                pass
            if i == max_tries - 1:
                logging.error(f"FAILED: {fn}")
                if os.path.exists(args[1]):
                    os.remove(args[1])
                return print(f"FAILED: {fn}")

    return wrapped


kwargs = {
    "format": "GTiff",
    "xRes": 0.00898315284120171538,
    "yRes": 0.00898315284120171538,
    "dstSRS": "EPSG:4326",
    "resampleAlg": "cubic",
    "creationOptions": [
        "TILED=YES",
        "COMPRESS=DEFLATE",
        "PREDICTOR=2",
        "BIGTIFF=YES",
    ],
}

ds_names = [
    "bdod",
    "cec",
    "cfvo",
    "clay",
    "nitrogen",
    "ocd",
    "ocs",
    "phh2o",
    "sand",
    "silt",
    "soc",
]

depths = ["0-5", "5-15", "15-30", "30-60", "60-100", "100-200"]

out_base_dir = "./data/soil"
curl_url = "/vsicurl?max_retry=3&retry_delay=5&list_dir=no&url="
base_url = "https://files.isric.org/soilgrids/latest/data/"
# ds_url = "ocs/ocs_0-30cm_mean.vrt"

args = []
for ds_name in ds_names:
    for depth in depths:
        ds_full_label = f"{ds_name}_{depth}cm_mean"

        out_dir = os.path.join(out_base_dir, ds_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_fn = os.path.join(out_dir, f"{ds_full_label}_1000m.tif")

        ds_url = f"{ds_name}/{ds_full_label}.vrt"
        url = curl_url + base_url + ds_url
        args.append((url, out_fn))


@retry
def warp(url, out_fn):
    gdal.Warp(out_fn, url, **kwargs)
    print(f"SUCCESS: {os.path.basename(out_fn)}.")
    logging.info(f"SUCCESS: {os.path.basename(out_fn)}.")


if __name__ == "__main__":
    with Pool(6) as pool:
        pool.starmap(warp, args)
