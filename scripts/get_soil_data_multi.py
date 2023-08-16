#!/usr/bin/env python3
import logging
import os
import time
from multiprocessing import Pool

from osgeo import gdal

RETRY_FAILED = True

logging.basicConfig(
    level=logging.INFO,
    filename="retry_failed_soil_data.log",
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


# def retry(f):
#     @wraps(f)
#     def wrapped(*args, **kwargs):
#         max_tries = 5
#         for i in range(max_tries):
#             fn = os.path.basename(args[1])
#             try:
#                 print(
#                     f"{bcolors.OKBLUE}Processing {bcolors.BOLD}{os.path.basename(fn)}{bcolors.ENDC}{bcolors.OKBLUE}... Attempt: {bcolors.ENDC}{i+1}"
#                 )
#                 return f(*args, **kwargs)
#             except Exception as ex:
#                 template = "An exception of type {0} occurred. Arguments:\n{1!r}"
#                 message = template.format(type(ex).__name__, ex.args)
#                 print(f"{bcolors.WARNING}{message}{bcolors.ENDC}")
#                 logging.warning(message)
#                 time.sleep(10)
#                 pass

#     return wrapped


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

out_base_dir = "./data/soil/1_km"
curl_url = "/vsicurl?max_retry=10&retry_delay=10&list_dir=no&url="
base_url = "https://files.isric.org/soilgrids/latest/data/"

if not RETRY_FAILED:
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

    args = []
    for ds_name in ds_names:
        for depth in depths:
            ds_full_label = f"{ds_name}_{depth}cm_mean"

            out_dir = os.path.join(out_base_dir, ds_name)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            out_fn = os.path.join(out_dir, f"{ds_full_label}_1_km.tif")

            ds_url = f"{ds_name}/{ds_full_label}.vrt"
            url = curl_url + base_url + ds_url
            args.append((url, out_fn))
else:
    failed = [
        "soc_60-100cm_mean",
    ]

    args = []
    for ds_full_label in failed:
        ds_name = ds_full_label.split("_")[0]

        out_dir = os.path.join(out_base_dir, ds_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        out_fn = os.path.join(out_dir, f"{ds_full_label}_1_km.tif")

        ds_url = f"{ds_name}/{ds_full_label}.vrt"
        url = curl_url + base_url + ds_url
        args.append((url, out_fn))


# @retry
def warp(url, out_fn):
    base = os.path.basename(out_fn)
    max_tries = 5
    ds = None
    for i in range(max_tries):
        try:
            print(
                f"{bcolors.OKBLUE}Processing {bcolors.BOLD}{base}{bcolors.ENDC}{bcolors.OKBLUE}... Attempt: {bcolors.ENDC}{i+1}"
            )
            ds = gdal.Warp(out_fn, url, **kwargs)
            break
        except Exception as ex:
            template = "An exception of type {0} occurred for {1}. Arguments:\n{2!r}"
            message = template.format(type(ex).__name__, ex.args, base)
            message = f"An exception of type {type(ex).__name__!r} occurred for {base}:\n{ex!s}"
            print(f"{bcolors.WARNING}{message}{bcolors.ENDC}")
            logging.warning(message)
            time.sleep(10)
            pass

    if ds is not None:
        print(f"{bcolors.OKGREEN}SUCCESS: {base}.{bcolors.ENDC}")
        logging.info(f"SUCCESS: {base}.")
    else:
        print(f"{bcolors.FAIL}FAILED: {base}{bcolors.ENDC}")
        logging.error(f"FAILED: {base}")

        if os.path.exists(out_fn):
            os.remove(out_fn)

    ds = None
    return ds


if __name__ == "__main__":
    gdal.UseExceptions()

    if not RETRY_FAILED:
        with Pool(12) as pool:
            pool.starmap(warp, args)
    else:
        for args_i in args:
            warp(*args_i)
