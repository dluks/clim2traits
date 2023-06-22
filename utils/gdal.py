################################
# GDAL utils
################################

import os
from functools import wraps
from typing import Callable, TypeVar

from osgeo import gdal
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")


def catch_gdal(f: Callable[P, T]) -> Callable[P, T]:
    """Wrapper function for error catching when using GDAL python API

    Args:
        f (Callable): Function that is executing the GDAL task

    Returns:
        Callable: The wrapped function
    """
    gdal.UseExceptions()

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

    @wraps(f)
    def wrapped(*args: P.args, **kwargs: P.kwargs) -> T:
        base = os.path.basename(str(kwargs["in_fn"]))
        try:
            return f(*args, **kwargs)
        except Exception as ex:
            template = "An exception of type {0} occurred for {1}. Arguments:\n{2!r}"
            message = template.format(type(ex).__name__, ex.args, base)
            message = f"An exception of type {type(ex).__name__!r} occurred for {base}:\n{ex!s}"
            print(f"{bcolors.WARNING}{message}{bcolors.ENDC}")

    return wrapped


@catch_gdal
def resample_gdal(
    in_fn: str,
    out_fn: str,
    res: float = 0.5,
    epsg: str = "EPSG:4326",
    globe: bool = False,
) -> gdal.Dataset:
    """Resamples a dataset to the desired resolution

    Args:
        in_fn (str): Input filepath
        out_fn (str): Output filepath
        res (float, optional): Desired square resolution in the units of the input
        dataset's CRS. Defaults to 0.5.
        epsg (str, optional): EPSG code of the desired output CRS
        globe (bool, optional): Update extent to full globe (only works if epsg = 4326)

    Returns:
        gdal.Dataset: Resampled dataset
    """

    kwargs = {
        "format": "GTiff",
        "xRes": res,
        "yRes": res,
        "dstSRS": epsg,
        "resampleAlg": "cubic",
        "multithread": True,
        "creationOptions": [
            "TILED=YES",
            "COMPRESS=DEFLATE",
            "PREDICTOR=2",
        ],
    }

    if globe and epsg == "EPSG:4326":
        kwargs["outputBounds"] = (-180, -90, 180, 90)

    ds = gdal.Warp(out_fn, in_fn, **kwargs)

    return ds
