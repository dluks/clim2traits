################################
# Geodata utils
################################
import json
import os
import re
import sys
from functools import reduce
from pathlib import Path
from typing import Optional, Tuple, Union

import dask_geopandas as dgpd
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as riox
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from tqdm import tqdm

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

    # Merge the two geodataframes on the geometry column such that only matching
    # geometries are retained
    merged = gpd.sjoin(grid1, grid2, how="inner", op="intersects")
    corr = merged[grid1_name].corr(merged[grid2_name])
    return corr


def compare_gdfs(gdf1: gpd.GeoDataFrame, gdf2: gpd.GeoDataFrame) -> float:
    """Calculate the correlation coefficient between two GeoDataFrames. Assumes that the
    two GeoDataFrames consist of a geometry column and a single data column.
    """
    col1 = gdf1.columns.difference(["geometry"]).values[0]
    col2 = gdf2.columns.difference(["geometry"]).values[0]
    merged = gpd.sjoin(gdf1, gdf2, how="inner", predicate="intersects")
    corr = merged[col1].corr(merged[col2])
    return corr


def compare_gdf_to_grid(
    gdf: gpd.GeoDataFrame,
    grid: xr.DataArray,
    gdf_name: str,
    grid_name: str,
) -> np.float64:
    """Calculate the correlation coefficient between a GeoDataFrame and a raster grid."""
    # Ensure that the grids share the same CRS
    grid = grid.rio.reproject("EPSG:4326")

    # Convert to GDFs
    grid = ds2gdf(grid, name=grid_name).dropna(subset=[grid_name])

    # Merge the two geodataframes on the geometry column such that only matching
    # geometries are retained
    merged = gpd.sjoin(gdf, grid, how="inner", predicate="intersects")

    corr = merged[gdf_name].corr(merged[grid_name])
    return corr


def splot_correlation_old(
    gdf: gpd.GeoDataFrame, trait_id: str, trait_name: str, splot_set: str = "orig"
) -> float:
    """Get the correlation between trait predictions and sPlot maps for a given trait.

    Args:
        gdf (gpd.GeoDataFrame): Trait predictions
        trait_id (str): Trait ID
        trait_name (str): Trait name
        splot_set (str, optional): sPlot dataset (one of ["orig", "05_range"], case-insensitive).

    Returns:
        float: Pearson correlation coefficient
    """
    splot = riox.open_rasterio(
        f"data/splot/0.01_deg/{splot_set.lower()}/sPlot_{trait_id}_0.01deg.tif",
        masked=True,
    )

    splot = ds2gdf(splot, trait_name)
    splot_corr = compare_gdfs(gdf, splot)

    return splot_corr


def splot_correlations(
    grid_res: Union[int, float], model_res: Union[int, float], pft: str
) -> Tuple[dict, dict]:
    """Get the correlations between trait predictions and sPlot maps for all traits in
    a given PFT and of a given resolution in degrees."""
    predictor_dataset = "MOD09GA.061_ISRIC_soil_WC_BIO_VODCA"
    nan_strat = "nan-strat=any_thr=0.5"
    dataset_name = f"{predictor_dataset}_{grid_res:g}_deg_{nan_strat}"

    dataset_dir_name = f"{'tiled_5x5_deg_' if grid_res == 0.01 else ''}{dataset_name}"

    prediction_dir = Path(
        "results/predictions",
        f"{num_to_str(model_res)}deg_models",
        dataset_dir_name,
        pft,
    )

    corr_table_gbif = {}
    corr_table_splot = {}

    for trait_dir in tqdm(list(prediction_dir.glob("TRYgapfilled*/GBIF"))):
        trait, corr = _trait_correlation(trait_dir, grid_res, pft)
        corr_table_gbif[trait] = corr

    for trait_dir in tqdm(list(prediction_dir.glob("TRYgapfilled*/sPlot"))):
        trait, corr = _trait_correlation(trait_dir, grid_res, pft)
        corr_table_splot[trait] = corr

    return corr_table_gbif, corr_table_splot


def _trait_correlation(
    trait_dir: Path, grid_res: Union[int, float], pft: str
) -> Tuple[str, float]:
    """Get the correlation between trait predictions and sPlot maps for a given trait."""
    trait = trait_dir.parent.name
    trait_id = get_trait_id_from_data_name(trait)
    trait_prediction_fn = list(trait_dir.glob("*.parq"))[0]

    if grid_res == 0.01:
        cols = dgpd.read_parquet(trait_prediction_fn).columns.values
        trait_prediction = dgpd.read_parquet(
            trait_prediction_fn, columns=cols[:2]
        ).compute()
    else:
        cols = gpd.read_parquet(trait_prediction_fn).columns.values
        trait_prediction = gpd.read_parquet(trait_prediction_fn, columns=cols[:2])

    # trait_prediction = trait_prediction.drop(
    #     columns=[
    #         "AOA",
    #         "DI",
    #         "CoV",
    #         *trait_prediction.columns[trait_prediction.columns.str.contains("masked")],
    #     ]
    # )

    if trait.endswith("_ln"):
        trait_prediction = back_transform_trait(trait_prediction)

    splot_dir = Path("GBIF_trait_maps/global_maps", pft, f"{num_to_str(grid_res)}deg")

    if grid_res >= 0.5:
        splot_fn = list(splot_dir.glob(f"*_X{trait_id}_*.grd"))[0]
        splot = riox.open_rasterio(splot_fn, masked=True).sel(band=2)
    else:
        splot_dir = splot_dir / "05_range"
        splot_fn = list(splot_dir.glob(f"*_X{trait_id}_*.tif"))[0]
        splot = riox.open_rasterio(splot_fn, masked=True).squeeze()

    if isinstance(splot, list):
        raise ValueError("Multiple sPlot files found.")

    splot = ds2gdf(splot, f"X{trait_id}").dropna(subset=[f"X{trait_id}"])
    corr = compare_gdfs(trait_prediction, splot)
    return f"X{trait_id}", corr


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

    encoding = {var: {"zlib": True, "complevel": 9} for var in ds.data_vars}

    ds.to_netcdf(out_fn, encoding=encoding)


def get_trait_id_from_data_name(data_name: str) -> str:
    """Get trait id from data name, e.g. GBIF_TRYgapfille   d_X1080_05deg_mean_ln -> 1080"""
    trait_id = re.search(r"X\d+", data_name).group()
    trait_id = trait_id.replace("X", "")
    return trait_id


def get_trait_name_from_trait_id(trait_id: str) -> str:
    """Get trait name from trait id, e.g. 1080 -> Root length per root dry mass
    (specific root length, SRL) (log-transformed)"""
    with open("./trait_mapping.json", encoding="utf-8") as f:
        mapping = json.load(f)
        trait_name = mapping[trait_id]["long"]
    return trait_name


def get_trait_name_from_data_name(data_name: str) -> str:
    """Get trait name from data name, e.g. GBIF_TRYgapfilled_X1080_05deg_mean_ln ->
    Root length per root dry mass (specific root length, SRL) (log-transformed)"""
    trait_id = get_trait_id_from_data_name(data_name)
    trait_name = get_trait_name_from_trait_id(trait_id)
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
