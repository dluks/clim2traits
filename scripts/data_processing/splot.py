import json
import logging
from pathlib import Path
from typing import Union

import geopandas as gpd
import pandas as pd
import xarray as xr
from geocube.api.core import make_geocube
from shapely.geometry import box, mapping

logging.basicConfig(level=logging.INFO)


def gdf_to_grid(
    gdf: gpd.GeoDataFrame,
    column: str,
    resolution: float = 0.01,
    bbox: tuple = (-180, -60, 180, 90),
) -> xr.DataArray:
    """Convert a GeoDataFrame to a xarray DataArray."""

    grid = make_geocube(
        vector_data=gdf,
        measurements=[column],
        resolution=(-resolution, resolution),
        geom=json.dumps(mapping(box(*bbox))),
    )

    return grid[column]


def write_grid(grid: xr.DataArray, out_path: Union[str, Path]) -> None:
    """Write a xarray DataArray to a NetCDF file."""

    grid.rio.to_raster(
        out_path,
        dtype="float32",
        compress="zstd",
        tiled=True,
        blockxsize=256,
        blockysize=256,
        predictor=2,
        num_threads=20,
        compute=False,
    )


def main():
    splot_cwm = (
        pd.read_csv("data/gbif-splot_raw/sPlotOpen_TRYgapfilled_cwm.csv")
        .set_index(["Longitude", "Latitude"])
        .groupby(["Longitude", "Latitude"])
        .mean()
        .drop(columns=["PlotObservationID", "Releve_area"])
        .reset_index()
    )

    splot_cwm = gpd.GeoDataFrame(
        splot_cwm,
        geometry=gpd.points_from_xy(splot_cwm.Longitude, splot_cwm.Latitude),
        crs="EPSG:4326",
    ).drop(columns=["Longitude", "Latitude"])

    traits = splot_cwm.columns.difference(["geometry"])

    for trait in traits:
        logging.info("Processing trait %s", trait)
        grid = gdf_to_grid(splot_cwm, trait)

        out_dir = Path("data/splot/0.01_deg")
        out_dir.mkdir(parents=True, exist_ok=True)

        write_grid(grid, out_dir / f"sPlot_TRYgapfilled_{trait}_0.01deg.tif")
        logging.info("Processed %s", trait)


if __name__ == "__main__":
    main()
