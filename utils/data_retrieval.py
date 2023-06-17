################################
# Data retrieval
################################
import glob
import itertools
import os

import geopandas as gpd

from utils.geodata import merge_dfs, tif2gdf


def get_fns(
    src_dir: str,
    res: str = "0.5_deg",
    ds: str = "inat",
    transform: str = "ln",
    bios: list[int] = None,
) -> list[str]:
    """Get dataset filenames by resolution and other dataset-specific identifiers

    Args:
        src_dir (str): Source dataset directory
        res (str, optional): Directory name of the desired resolution. Defaults to
        "0.5_deg".
        ds (str, optional): Dataset name, case-insensitive. Defaults to "inat".
        transform (str, optional): Transform (only applies to iNat data). Defaults to
        "ln".
        bios (list[int], optional): Bio variables (only applies to WorldClim data).
        Defaults to None.

    Raises:
        ValueError: ds must be one of ["inat", "modis", "wc", "soil"]

    Returns:
        list[str]: List of filenames
    """
    ds = ds.lower()
    valid = ["inat", "modis", "wc", "soil"]
    if ds not in valid:
        raise ValueError(f"Dataset (ds) must be one of {valid:r} (case-insensitive)")

    # iNaturalist
    if ds == "inat":
        fns = glob.glob(os.path.join(src_dir, f"{res}/{transform}", "*.tif"))

    # WorldClim bio variables
    elif ds == "wc":
        wc_dir = os.path.join(src_dir, res)

        if not bios:
            fns = glob.glob(os.path.join(wc_dir, "*.tif"))
        else:
            fns = list(
                itertools.chain.from_iterable(
                    [
                        glob.glob(f"{wc_dir}/wc2.1_{res}_bio_{b}.tif", recursive=True)
                        for b in bios
                    ]
                )
            )

    # MODIS
    elif ds == "modis":
        fns = glob.glob(os.path.join(src_dir, res, "*.tif"))

    # Soil
    elif ds == "soil":
        fns = glob.glob(os.path.join(src_dir, "*", res, "*.tif"))

    return sorted(fns)


def gdf_from_list(fns: list[str]) -> gpd.GeoDataFrame:
    """Get rasters from a list, convert to GeoDataFrames, and merge them into a single
    GeoDataFrame.

    Args:
        fns (list[str]): List of xr.DataArrays or filenames

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame
    """
    gdfs = []
    for fn in fns:
        gdfs.append(tif2gdf(fn))

    merged_gdfs = merge_dfs(gdfs)
    return merged_gdfs
