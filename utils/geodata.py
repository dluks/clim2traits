################################
# Geodata utils
################################
import os
from functools import reduce
from typing import Optional, Tuple, Union

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling

NPARTITIONS = os.cpu_count()


def ds2gdf(ds: xr.DataArray, name: Optional[str] = None) -> gpd.GeoDataFrame:
    """Converts an xarray dataset to a geopandas dataframe

    Args:
        ds (xarray.DataArray): Dataset to convert
        name (Optional[str], optional): Name of the dataset. Defaults to None.

    Returns:
        geopandas.GeoDataFrame: GeoPandas dataframe
    """
    ds_name = name if name is not None else ds.name

    df = ds.to_dataframe(name=ds_name).reset_index()
    geometry = gpd.points_from_xy(df.x, df.y)
    df = df.drop(columns=["band", "spatial_ref", "x", "y"])
    gdf = gpd.GeoDataFrame(data=df, crs=ds.rio.crs, geometry=geometry)

    return gdf


def tif2gdf(
    raster: Union[str, xr.DataArray], band_id: Optional[int] = None
) -> gpd.GeoDataFrame:
    """Converts a GeoTIFF to a GeoPandas data frame. Accepts either the filename of a
    GeoTiff or an xr dataset.

    Args:
        raster (str || xarray.DataArray): Filename of the GeoTIFF raster or the opened
            raster itself.
        band (Optional[int], optional): Band number. Defaults to None.

    Returns:
        geopandas.GeoDataFrame: GeoPandas data frame
    """
    ds = validate_raster(raster)
    band_gdfs = []

    if band_id:
        if "long_name" in ds.attrs.keys():
            band_name = ds.attrs["long_name"][band_id - 1]  # Band ints aren't 0-indexed
        else:
            band_name = f"_band{band_id:02d}"
        band_ds = ds.sel(band=band_id)
        band_ds.name = f"{ds.name}_{band_name}"
        band_gdfs.append(ds2gdf(band_ds))
    else:
        for band in ds.band.values:
            band_ds = ds.sel(band=band)
            if ds.band.values.size > 1:
                band_ds.name = f"{ds.name}_band{band:02d}"
            band_gdfs.append(ds2gdf(band_ds))

    if len(band_gdfs) > 1:
        gdf = merge_gdfs(band_gdfs)
    elif len(band_gdfs) == 0:
        raise ValueError("No bands found in raster.")
    else:
        gdf = band_gdfs[0]

    return gdf


def merge_gdfs(
    gdfs: list[gpd.GeoDataFrame], method: str = "original"
) -> gpd.GeoDataFrame:
    """Merge GeoDataFrames on matching data

    Args:
        gdfs (list[gpd.GeoDataFrame]): List of GeoDataFrames to merge
            on a common column
        method (str, optional): Expected geometries across gdfs. Defaults to
            "original".

    Returns:
        gpd.GeoDataFrame: Merged DataFrame
    """
    crs = gdfs[0].crs

    if method == "original":
        merged_gdf = reduce(lambda left, right: pd.merge(left, right, how="left"), gdfs)
    elif method == "matching":
        # Check that all geometries are the same
        for gdf in gdfs:
            if not np.array_equal(gdf.geometry.values, gdfs[0].geometry.values):
                raise ValueError("Geometries do not match across GeoDataFrames.")

        geometry = gdfs[0].geometry
        dfs = [gdf.drop(columns=["geometry"]) for gdf in gdfs]

        merged_gdf = gpd.GeoDataFrame(
            dfs[0].join(dfs[1:], how="left"), crs=crs, geometry=geometry
        )
    elif method == "unique":
        geometry = [g for gdf in gdfs for g in gdf.geometry]
        dfs = [gdf.drop(columns=["geometry"]) for gdf in gdfs]
        merged_gdf = gpd.GeoDataFrame(
            pd.concat(dfs, ignore_index=True).reset_index(), crs=crs, geometry=geometry
        )

    return merged_gdf


def resample_raster(
    dataset: xr.DataArray, upscale_factor: float = 1 / 3
) -> xr.DataArray:
    """Resample a rioxarray dataset according to an upscale factor.

    Args:
        dataset (xarray.DataArray): Raster dataset
        upscale_factor (float, optional): Scaling factor. Upsamples if between 0 and 1.
        Defaults to 1/3.

    Returns:
        xarray.DataArray: Resampled dataset
    """
    height = int(dataset.rio.height * upscale_factor)
    width = int(dataset.rio.width * upscale_factor)
    resampled = dataset.rio.reproject(
        dataset.rio.crs, shape=(height, width), resampling=Resampling.bilinear
    )

    return resampled


def print_shapes(
    X: gpd.GeoDataFrame, Y: gpd.GeoDataFrame, rows_dropped: bool = True
) -> None:
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    if rows_dropped:
        rows = abs((Y.shape[0] - X.shape[0]))
        if rows == 0:
            print("Rows match\n")
        else:
            print("Rows dropped:", rows, "\n")


def drop_XY_NAs(
    XY: Union[gpd.GeoDataFrame, pd.DataFrame],
    X_cols: Union[pd.Index, str],
    Y_cols: Union[pd.Index, str],
    verbose: int = 0,
) -> Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.Index, Union[pd.Index, str]]:
    """Drop all rows and columns that contain all NAs in either X (predictors) or Y
    (response variables).

    Args:
        XY (Union[gpd.GeoDataFrame, pd.DataFrame]): Dataframe containing both X and Y
            variables
        X_cols (pd.Index): Index identifying the column(s) containing the predictors
        Y_cols (pd.Index, str): Index or string identifying the column(s) containing the response variable(s)
        verbose (int, optional): Verbosity level. Defaults to 0.

    Returns:
        Tuple[Union[gpd.GeoDataFrame, pd.DataFrame], pd.Index, pd.Index]: Original
            dataframe with full-NA rows/columns in the X and Y spaces dropped.
    """

    shape_before = XY.shape

    # Drop all rows that contain no response variable data at all
    XY = XY.dropna(axis=0, subset=Y_cols, how="all")
    # Drop all rows that contain no predictor data at all
    XY = XY.dropna(axis=0, subset=X_cols, how="all")
    # Drop all columns that contain no data at all
    XY = XY.dropna(axis=1, how="all")

    if (isinstance(Y_cols, str)) and (Y_cols not in XY.columns):
        raise ValueError(f"The single y column ({Y_cols}) contained all NAs!")

    if verbose:
        print(f"XY shape before dropping full-NA rows/cols: {shape_before}")
        shape_after = XY.shape
        rows_dropped = shape_before[0] - shape_after[0]
        message = ""
        message += f"XY shape after:                             {shape_after}"
        message += f"\n# of rows excluded: {rows_dropped} ({(rows_dropped / shape_before[0]) * 100:.2f}%)"
        dropped_X_cols = X_cols[~X_cols.isin(XY.columns)].values

        if (isinstance(Y_cols, str)) and (Y_cols in XY.columns):
            dropped_Y_cols = []
        else:
            dropped_Y_cols = Y_cols[~Y_cols.isin(XY.columns)].values

        if len(dropped_X_cols) == 0 and len(dropped_Y_cols) == 0:
            message += "\n\nNo columns were dropped."
        else:
            message += "\nEmpty columns:"
            message += f"\nX: {dropped_X_cols}"
            message += f"\nY: {dropped_Y_cols}"

        print(message)
    # Update X_cols and Y_cols accordingly to account for any dropped columns
    X_cols = X_cols[X_cols.isin(XY.columns)]
    Y_cols = Y_cols[Y_cols.isin(XY.columns)] if not isinstance(Y_cols, str) else Y_cols

    return XY, X_cols, Y_cols


def get_epsg(raster: Union[str, os.PathLike, xr.DataArray]) -> int:
    """Get the EPSG code of a raster.

    Args:
        raster (Union[str, os.PathLike, xr.DataArray]): Raster dataset

    Returns:
        int: EPSG code
    """
    dataset = validate_raster(raster)

    return int(dataset.rio.crs.to_epsg())


def validate_raster(raster: Union[str, os.PathLike, xr.DataArray]) -> xr.DataArray:
    """Validate a raster dataset.

    Args:
        raster (Union[str, os.PathLike, xr.DataArray]): Raster dataset

    Returns:
        xr.DataArray: Validated raster dataset
    """
    if isinstance(raster, (str, os.PathLike)):
        name = os.path.splitext(os.path.basename(raster))[0]
        dataset = riox.open_rasterio(raster, masked=True, default_name=name)
        if isinstance(dataset, list):
            dataset = dataset[0]
        return dataset

    if not isinstance(raster, xr.DataArray):
        raise TypeError("Raster is neither a filename nor a valid dataset.")

    return raster


def netcdf2gdf(da: xr.DataArray, data_label: str, name: str) -> gpd.GeoDataFrame:
    """Converts a netCDF dataset to a GeoDataFrame"""
    # Convert to a DataFrame
    df = da.to_dataframe().reset_index()
    df = df.rename(columns={"lon": "x", "lat": "y"})
    df = df.dropna(subset=["x", "y"])
    if "months" in df.columns:
        df = df.drop(columns=["month"])
    df = df.rename(columns={data_label: name})
    geometry = gpd.points_from_xy(df.x, df.y)
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df = df.set_crs("EPSG:4326")
    return df


def ts_netcdf2gdfs(
    ds: Union[xr.Dataset, str, os.PathLike], ds_name: Optional[str] = None
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Converts a timeseries netCDF dataset to a GeoDataFrame with columns for each time step.

    Args:
        ds (Union[xr.Dataset, str, os.PathLike]): Dataset
        ds_name (Optional[str], optional): Name of the dataset. Defaults to None.

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: GeoDataFrame with columns for each time step

    """

    gdfs = []

    if isinstance(ds, (str, os.PathLike)):
        band_name = str(ds).split("_")[0].lower() + "_band"
        ds = xr.open_dataset(ds)

    data_label = str(list(ds.keys())[0])  # assume that the first variable is the data

    if data_label == "Band1":
        # In this case, the GDAL resampling resulted in the "month" dimension being
        # expanded into 12 separate data variables. This could be cleaned up as a TODO.
        for i in range(1, 13):
            data_label = f"Band{i}"
            da = ds[data_label]

            if ds_name:
                da_name = f"{ds_name}_{band_name}_m{i:02d}"
            else:
                da_name = f"{data_label}_m{i:02d}"
            df = netcdf2gdf(da, data_label, da_name)
            gdfs.append(df)
    else:
        for da in ds[data_label]:
            month = int(da.month.values)

            if ds_name:
                da_name = f"{ds_name}_{data_label}_m{month:02d}"
            else:
                da_name = f"{data_label}_m{month:02d}"

            df = netcdf2gdf(da, data_label, da_name)
            gdfs.append(df)

    if len(gdfs) > 1:
        gdf = merge_gdfs(gdfs)
    else:
        gdf = gdfs[0]

    return gdf


def mask_oceans(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Masks oceans from a GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame with oceans masked.
    """
    land_mask = gpd.read_feather("./data/masks/land_mask_110m.feather")
    land_mask = land_mask.to_crs(gdf.crs)
    # if isinstance(gdf, dgpd.GeoDataFrame):
    #     gdf = gdf.compute()
    return gdf.clip(land_mask)


def clip_to_land(ds: xr.Dataset) -> xr.Dataset:
    """Clips a raster dataset to land only."""
    land_mask = gpd.read_feather("./data/masks/land_mask_110m.feather")
    if not ds.rio.crs:
        ds.rio.write_crs("EPSG:4326", inplace=True)

    ds = ds.rio.clip(
        geometries=land_mask.geometry.values,
        crs=land_mask.crs,
        all_touched=False,
        drop=False,
        invert=False,
    )

    return ds


def pad_raster(ds: xr.Dataset) -> xr.Dataset:
    """Pads a raster dataset to the full extent of the globe."""
    ds = ds.rio.pad_box(minx=180, miny=-90, maxx=180, maxy=90)
    return ds


def compare_grids(
    grid1: xr.DataArray, grid2: xr.DataArray, grid1_name: str, grid2_name: str
) -> np.float64:
    """Calculate the correlation coefficient between two raster grids."""
    # Ensure that the grids share the same CRS
    grid1 = grid1.rio.reproject("EPSG:4326")
    grid2 = grid2.rio.reproject("EPSG:4326")

    # Convert to GDFs
    grid1 = ds2gdf(grid1, name=grid1_name).dropna(subset=[grid1_name])
    grid2 = ds2gdf(grid2, name=grid2_name).dropna(subset=[grid2_name])

    # Merge the two geodataframes on the geometry column such that only matching geometries are retained
    merged = gpd.sjoin(grid1, grid2, how="inner", op="intersects")
    corr = merged[grid1_name].corr(merged[grid2_name])
    return corr


def compare_gdf_to_grid(
    gdf: gpd.GeoDataFrame, grid: xr.DataArray, gdf_name: str, grid_name: str
) -> np.float64:
    """Calculate the correlation coefficient between a GeoDataFrame and a raster grid."""
    # Ensure that the grids share the same CRS
    grid = grid.rio.reproject("EPSG:4326")

    # Convert to GDFs
    grid = ds2gdf(grid, name=grid_name).dropna(subset=[grid_name])

    # Merge the two geodataframes on the geometry column such that only matching geometries are retained
    merged = gpd.sjoin(gdf, grid, how="inner", op="intersects")
    corr = merged[gdf_name].corr(merged[grid_name])
    return corr
