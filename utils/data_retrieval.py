################################
# Data retrieval
################################
from __future__ import annotations

from pathlib import Path
from typing import Optional

import geopandas as gpd

from utils.dataset_tools import FileExt
from utils.geodata import merge_dfs, tif2gdf, ts_netcdf2gdfs


def gdf_from_list(
    fns: list[str], ds_name: Optional[str] = None, band: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Get rasters from a list, convert to GeoDataFrames, and merge them into a single
    GeoDataFrame.

    Args:
        fns (list[str]): List of xr.DataArrays or filenames
        ds_name (Optional[str], optional): Dataset name. Defaults to None.
        band (Optional[int], optional): Band number. Defaults to None.

    Returns:
        gpd.GeoDataFrame: Combined GeoDataFrame
    """
    gdfs = []

    for fn in fns:
        file_ext = FileExt(Path(fn).suffix[1:])
        if file_ext == FileExt.TIF or file_ext == FileExt.GRID:
            gdfs.append(tif2gdf(fn, band_id=band))
        elif file_ext == FileExt.NETCDF4:
            gdfs.append(ts_netcdf2gdfs(fn, ds_name))

    merged_gdfs = merge_dfs(gdfs)

    return merged_gdfs
