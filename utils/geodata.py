################################
# Geodata utils
################################
import os
from functools import reduce
from typing import Optional, Tuple, Union

import geopandas as gpd
import pandas as pd
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling


def ds2gdf(ds: xr.DataArray) -> gpd.GeoDataFrame:
    """Converts an xarray dataset to a geopandas dataframe

    Args:
        ds (xarray.DataArray): Dataset to convert

    Returns:
        geopandas.GeoDataFrame: GeoPandas dataframe
    """
    df = ds.to_dataframe().reset_index()
    geometry = gpd.points_from_xy(df.x, df.y)
    gdf = gpd.GeoDataFrame(data=df, crs=ds.rio.crs, geometry=geometry)
    gdf = gdf.drop(columns=["band", "spatial_ref"])

    return gdf


def tif2gdf(raster: Union[str, xr.DataArray]) -> gpd.GeoDataFrame:
    """Converts a GeoTIFF to a GeoPandas data frame. Accepts either the filename of a
    GeoTiff or an xr dataset.

    Args:
        raster (str || xarray.DataArray): Filename of the GeoTIFF raster or the opened
        raster itself.

    Returns:
        geopandas.GeoDataFrame: GeoPandas data frame
    """
    ds = validate_raster(raster)

    band_gdfs = []

    for band in ds.band.values:
        band_ds = ds.sel(band=band)
        if ds.band.values.size > 1:
            band_ds.name = f"{ds.name}_band{band:02d}"
        band_gdfs.append(ds2gdf(band_ds))
        # band_df = band_ds.to_dataframe().reset_index()
        # geometry = gpd.points_from_xy(band_df.x, band_df.y)
        # band_gdf = gpd.GeoDataFrame(data=band_df, crs=ds.rio.crs, geometry=geometry)
        # band_gdfs.append(band_gdf)

    if len(band_gdfs) > 1:
        gdf = merge_dfs(band_gdfs)
    elif len(band_gdfs) == 0:
        raise ValueError("No bands found in raster.")
    else:
        gdf = band_gdfs[0]

    return gdf


def merge_dfs(
    gdfs: list[Union[pd.DataFrame, gpd.GeoDataFrame]],
    how: str = "left",
) -> Union[pd.DataFrame, gpd.GeoDataFrame]:
    """Merge GeoDataFrames on matching data

    Args:
        gdfs (list[Union[pd.DataFrame, gpd.GeoDataFrame]]): List of DataFrames to merge
            on a common column
        how (str, optional): Type of merge to be performed. Defaults to "inner".

    Returns:
        Union[pd.DataFrame, gpd.GeoDataFrame]: Merged DataFrame
    """
    merged_gdf = reduce(lambda left, right: pd.merge(left, right, how=how), gdfs)

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


def validate_raster(
    raster: Union[str, os.PathLike, xr.DataArray]
) -> Union[xr.Dataset, xr.DataArray]:
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
    df = da.to_dataframe()
    df = df.reset_index()
    df = df.rename(columns={"lon": "x", "lat": "y"})
    df = df.dropna(subset=["x", "y"])
    df = df.drop(columns=["time"])
    df = df.rename(columns={data_label: name})
    geometry = gpd.points_from_xy(df.x, df.y)
    df = gpd.GeoDataFrame(df, geometry=geometry)
    df.set_crs(epsg="4326", inplace=True)
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
        ds = xr.open_dataset(ds)

    data_label = str(list(ds.keys())[0])  # assume that the first variable is the data

    for da in ds[data_label]:
        date = pd.to_datetime(str(da.time.values)).strftime("%Y-%m-%d")

        da_name = (
            f"{ds_name}_{data_label}_{date}" if ds_name else f"{data_label}_{date}"
        )
        df = netcdf2gdf(da, data_label, da_name)
        gdfs.append(df)

    if len(gdfs) > 1:
        gdf = merge_dfs(gdfs)
    else:
        gdf = gdfs[0]

    return gdf
