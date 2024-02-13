################################
# Geodata utils
################################
import json
import os
import re
import sys
from functools import reduce
from pathlib import Path
from typing import Generator, Iterator, List, Optional, Tuple, Union

import dask.dataframe as dd
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as riox
import xarray as xr
from numpy.typing import ArrayLike
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm

os.environ["USE_PYGEOS"] = "0"

import dask_geopandas as dgpd
import geopandas as gpd

NPARTITIONS = os.cpu_count()


def ds2gdf(
    ds: xr.DataArray, name: Optional[str] = None
) -> Union[gpd.GeoDataFrame, dgpd.GeoDataFrame]:
    """Converts an xarray dataset to a geopandas dataframe

    Args:
        ds (xarray.DataArray): Dataset to convert
        name (Optional[str], optional): Name of the dataset. Defaults to None.

    Returns:
        geopandas.GeoDataFrame: GeoPandas dataframe
    """
    ds_name = name if name is not None else ds.name
    crs = "EPSG:4326"

    out = ds.to_dataframe(name=ds_name).reset_index()
    out = out.drop(columns=["band", "spatial_ref"])
    out = out.dropna(subset=out.columns.difference(["x", "y"]))

    geometry = gpd.points_from_xy(out.x, out.y)
    out = out.drop(columns=["x", "y"])
    out = gpd.GeoDataFrame(data=out, crs=crs, geometry=geometry)

    return out


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

    if band_id and "band" in ds.coords:
        if "long_name" in ds.attrs.keys():
            band_name = ds.attrs["long_name"][band_id - 1]  # Band ints aren't 0-indexed
        else:
            band_name = f"_band{band_id:02d}"

        if band_id not in ds.band.values:
            # Should raise an error here, but this is a quick fix for now
            band_gdfs.append(ds2gdf(ds))
        else:
            band_ds = ds.sel(band=band_id)
            band_ds.name = f"{ds.name}_{band_name}"
            band_gdfs.append(ds2gdf(band_ds))
    elif "band" in ds.coords:
        for band in ds.band.values:
            band_ds = ds.sel(band=band)
            if ds.band.values.size > 1:
                band_ds.name = f"{ds.name}_band{band:02d}"
            band_gdfs.append(ds2gdf(band_ds))
    else:
        band_gdfs.append(ds2gdf(ds))

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
    """Print the shapes of the X and Y dataframes."""
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
        Y_cols (pd.Index, str): Index or string identifying the column(s) containing the
            response variable(s)
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
        band_name = str(ds).split("_", maxsplit=1)[0].lower() + "_band"
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


def fill_holes(ds: xr.Dataset, method: str = "cubic") -> xr.Dataset:
    """Fills holes in a dataset."""
    ds = ds.rio.interpolate_na(method=method)
    ds = clip_to_land(ds)
    ds = pad_ds(ds)
    return ds


def pad_ds(
    ds: xr.Dataset, minx: int = -180, miny: int = -60, maxx: int = 180, maxy: int = 90
) -> xr.Dataset:
    """Pads a raster dataset to the full extent of the globe."""
    ds = ds.rio.pad_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
    return ds


def back_transform_trait(gdf: gpd.GeoDataFrame, drop: bool = True) -> gpd.GeoDataFrame:
    """Back-transforms the log-transformed trait values in the given GeoDataFrame.

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame containing log-transformed trait values
        drop (bool, optional): Whether to drop the log-transformed trait column. Defaults to True.

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing back-transformed trait values
    """
    log_col_name = gdf.columns.difference(["geometry"]).values[0]
    bt_col_name = log_col_name.replace("_ln", "")

    gdf[bt_col_name] = np.exp(gdf[log_col_name])

    if drop:
        gdf = gdf.drop(columns=[log_col_name])

    return gdf


def read_001_predictions(trait_id: str, model: str = "GBIF") -> gpd.GeoDataFrame:
    """Reads the 0.01 degree predictions for a given trait and model.

    Args:
        trait_id (str): Trait ID
        model (str, optional): . Model (one of ["GBIF", "sPlot"], case-insensitive).
            Defaults to "GBIF".

    Returns:
        gpd.GeoDataFrame: GeoDataFrame containing the predictions
    """
    model = "GBIF" if model.lower() == "gbif" else "sPlot"

    log = "_ln" if trait_id == "X50" else ""
    col = f"{model}_TRYgapfilled_{trait_id}_05deg_mean{log}"
    pred_path = Path(
        "results",
        "predictions",
        "tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_nan-strat=any_thr=0.5",
        f"TRYgapfilled_{trait_id}_05deg_mean{log}",
        model,
        "merged_predictions.parq",
    )

    gdf = gpd.read_parquet(
        pred_path,
        columns=["geometry", col],
    )

    if trait_id == "X50":
        gdf = back_transform_trait(gdf)

    return gdf


def num_to_str(number: Union[int, float]) -> str:
    """Converts a number to a string with a leading zero if the number is less than 0.
    Decimals are removed. Negative numbers are converted to positive numbers before
    conversion.
    E.g. 1.0 -> 1, 0.5 -> 05, 0.001 -> 001, -1 -> 1"""
    return f"{np.abs(number):g}".replace(".", "")


def ds_to_gtiff(
    ds: Union[xr.Dataset, xr.DataArray], out_fn: Union[str, os.PathLike]
) -> None:
    """Writes a rioxarray dataset to a raster."""
    # Ensure that dataset is geospatial
    try:
        ds.rio.resolution()
    except AttributeError as e:
        print(e)
        sys.exit()

    assert ds.rio.crs is not None, "Dataset does not have a CRS."

    write_args = {
        "tiled": True,
        "compress": "zstd",
        "predictor": 2,
        "num_threads": os.cpu_count(),
        "windowed": False,
        "compute": False,
    }

    ds.rio.to_raster(
        out_fn,
        tags={
            "compression": "zstd",
            "compression_predictor": 2,
            "resolution": 0.01,
            "resolution_unit": "degree",
        },
        **write_args,
    )


def ds_to_netcdf(
    ds: Union[xr.Dataset, xr.DataArray], out_fn: Union[str, os.PathLike]
) -> None:
    """Writes a rioxarray dataset to a netCDF."""
    # Ensure that dataset is geospatial
    try:
        ds.rio.resolution()
    except AttributeError as e:
        print(e)
        sys.exit()

    if ds.rio.crs is None:
        raise ValueError("Dataset does not have a CRS.")

    encoding = {var: {"zlib": True, "complevel": 9} for var in ds.data_vars}

    mode = "w"

    if Path(out_fn).exists():
        mode = "a"

    ds.to_netcdf(out_fn, mode=mode, encoding=encoding)


def get_trait_id_from_data_name(data_name: str) -> Union[str, None]:
    """Get trait id from data name, e.g. GBIF_TRYgapfilled_X1080_05deg_mean_ln -> 1080"""
    trait_id = re.search(r"X\d+", data_name)
    if trait_id is not None:
        return trait_id.group().replace("X", "")
    return None


def get_trait_name_from_trait_id(trait_id: str, short: bool = False) -> str:
    """Get trait name from trait id, e.g. 1080 -> Root length per root dry mass
    (specific root length, SRL) (log-transformed)"""
    with open("./trait_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
        if short:
            trait_name = mapping[trait_id]["short"]
        else:
            trait_name = mapping[trait_id]["long"]
    return trait_name


def get_trait_name_from_data_name(data_name: str, short: bool = False) -> str:
    """Get trait name from data name, e.g. GBIF_TRYgapfilled_X1080_05deg_mean_ln ->
    Root length per root dry mass (specific root length, SRL) (log-transformed)"""
    trait_id = get_trait_id_from_data_name(data_name)
    trait_name = get_trait_name_from_trait_id(trait_id, short=short)
    return trait_name


def rasterize_points(
    points: np.ndarray, res: Union[int, float], bbox: tuple
) -> np.ndarray:
    """Rasterize points into a grid with a given resolution.

    Args:
        points (np.ndarray): Points to rasterize, with columns (x, y, value) (for
            geographic coordinates, use (lon, lat, value))
        res (Union[int, float]): Resolution of the grid
        bbox (tuple): Bounding box of the grid

    Returns:
        np.ndarray: Rasterized grid
    """
    width = int((bbox[2] - bbox[0]) / res)
    height = int((bbox[3] - bbox[1]) / res)

    rast = np.zeros((height, width), dtype=np.float32)
    count_array = np.zeros_like(rast)

    for x, y, value in tqdm(points):
        col = int((x - bbox[0]) / res)
        row = int((bbox[3] - y) / res)
        rast[row, col] += value
        count_array[row, col] += 1

    # Avoid division by zero
    count_array[count_array == 0] = 1

    # Calculate the average
    rast = rast / count_array

    return rast


def write_raster(
    raster: np.ndarray, res: Union[int, float], bbox: tuple, filename: os.PathLike
) -> None:
    """Write a raster to a GeoTIFF file.

    Args:
        raster (np.ndarray): Raster matrix to write
        res (Union[int, float]): Resolution of the raster
        bbox (tuple): Bounding box of the raster
        filename (os.PathLike): Path to the output file
    """
    width = int((bbox[2] - bbox[0]) / res)
    height = int((bbox[3] - bbox[1]) / res)

    with rio.open(
        filename,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        nodata=0,
        count=1,
        dtype=rio.float32,
        crs="EPSG:4326",
        transform=from_origin(bbox[0], bbox[3], res, res),
        compress="zstd",
    ) as dst:
        dst.write(raster, 1)


def pack_ds(ds: xr.Dataset) -> xr.Dataset:
    """Pack a dataset to save memory."""

    if ds.rio.crs is None:
        raise ValueError("Dataset does not have a CRS.")

    NODATA = -(2**15)

    with xr.set_options(keep_attrs=True):
        for dv in ds.data_vars:
            if np.issubdtype(ds[dv].dtype, np.floating):
                min_val = ds[dv].min().item()
                max_val = ds[dv].max().item()

                scale_factor = (max_val - min_val) / (2**16 - 2)
                scale_factor = np.array([scale_factor]).astype(np.float32)[0]

                offset = (max_val + min_val) / 2
                offset = np.array([offset]).astype(np.float32)[0]

                ds[dv] = (ds[dv] - offset) / scale_factor
                ds[dv] = ds[dv].fillna(NODATA)
                ds[dv] = ds[dv].rio.write_nodata(NODATA, encoded=True)

                ds[dv] = ds[dv].astype(np.int16)

                ds[dv].attrs["scale_factor"] = scale_factor
                ds[dv].attrs["add_offset"] = offset
                ds[dv].attrs["_FillValue"] = NODATA

    return ds


def da_to_ds(da: xr.DataArray, name: Optional[str] = None) -> xr.Dataset:
    """Convert a DataArray to Dataset with appropriately-named data variables."""
    assert "long_name" in da.attrs, "long_name attribute is required"

    crs = da.rio.crs

    if "band" in da.dims:
        ds = da.to_dataset(dim="band")
        ds = ds.rename_vars(
            {dv: ds.attrs["long_name"][i] for i, dv in enumerate(ds.data_vars)}
        )
    else:
        if name is None:
            raise ValueError("name is required if DataArray has no band dimension.")
        ds = da.to_dataset(name=name)

    with xr.set_options(keep_attrs=True):
        for dv in ds.data_vars:
            ds[dv].attrs["long_name"] = str(dv)

    ds = ds.rio.write_crs(crs)
    ds.attrs["crs"] = crs.to_string()
    ds.attrs["long_name"] = [str(dv) for dv in ds.data_vars]

    return ds


def open_raster(
    filename: Union[str, os.PathLike], **kwargs
) -> Union[xr.DataArray, xr.Dataset]:
    """Open a raster dataset using rioxarray."""
    ds = riox.open_rasterio(filename, **kwargs)

    if isinstance(ds, list):
        raise ValueError("Multiple files found.")

    return ds


def get_prediction_paths(
    resolution: Union[int, float], pft: str, trait_idx: Optional[List[int]] = None
) -> Iterator[Path]:
    """Get the paths to the predictions for a given resolution and PFT."""
    if resolution == 0.01:
        predict_name = (
            "tiled_5x5_deg_MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.01_deg_nan-strat=any_"
            "thr=0.5"
        )
    else:
        predict_name = (
            f"MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_{resolution:g}_deg_nan-strat=any_"
            "thr=0.5"
        )

    if trait_idx is not None:
        for p in Path(
            f"results/predictions/05deg_models/{predict_name}/{pft}"
        ).iterdir():
            tid = get_trait_id_from_data_name(p.name)
            if tid is None:
                continue
            if int(tid) in trait_idx:
                yield p
    else:
        return Path(f"results/predictions/05deg_models/{predict_name}/{pft}").glob("*")


def test():
    for trait in tqdm(predicted_traits):
        if not trait.is_dir():
            continue
        gbif_fn = list(Path(trait, "GBIF").glob("*.parq"))[0]
        splot_fn = list(Path(trait, "sPlot").glob("*.parq"))[0]

        if dask:
            columns = dd.read_parquet(gbif_fn).columns.values
            if "AOA" not in columns:
                continue
            else:
                gbif_trait_df = dd.read_parquet(gbif_fn, columns=["AOA"])
                splot_trait_df = dd.read_parquet(splot_fn, columns=["AOA"])
        else:
            columns = pd.read_parquet(gbif_fn).columns.values
            if "AOA" not in columns:
                continue
            else:
                gbif_trait_df = pd.read_parquet(gbif_fn, columns=["AOA"])
                splot_trait_df = pd.read_parquet(splot_fn, columns=["AOA"])
