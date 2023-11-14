import logging
import os
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
import rasterio as rio
import rioxarray as riox
from rasterio.transform import from_origin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

### CONFIG
RANGE_05 = False


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
    for x, y, value in points:
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


def main():
    """Convert sPlot data to raster grids."""
    splot_cwm = pd.read_csv("data/gbif-splot_raw/sPlotOpen_TRYgapfilled_cwm.csv").drop(
        columns=["PlotObservationID", "Releve_area"]
    )

    out_dir = Path("data/splot/0.01_deg/")
    out_dir.mkdir(parents=True, exist_ok=True)

    traits = splot_cwm.columns.difference(["Latitude", "Longitude"])

    res = 0.01
    bbox = (-180, -60, 180, 90)

    for trait in traits:
        logging.info("Processing trait %s", trait)

        points = (
            splot_cwm[["Longitude", "Latitude", trait]].dropna(subset=[trait]).values
        )

        if RANGE_05:
            out_dir = out_dir / "05_range"
            splot_05 = riox.open_rasterio(
                Path(
                    "GBIF_trait_maps/global_maps/Shrub_Tree_Grass/05deg/",
                    f"sPlot_TRYgapfilled_{trait}_05deg.grd",
                ),
                masked=True,
            )

            # Get min and max values of splot_05
            min_val = splot_05.min().values.item()
            max_val = splot_05.max().values.item()

            # Drop points outside min_val and max_val, and store count of number of points dropped
            before = points.shape[0]
            points = points[
                np.logical_and(points[:, 2] >= min_val, points[:, 2] <= max_val)
            ]
            after = points.shape[0]
            logging.info("Dropped %d points outside range", before - after)
        else:
            out_dir = out_dir / "orig"

        grid = rasterize_points(points, res, bbox)
        write_raster(grid, res, bbox, out_dir / f"sPlot_{trait}_0.01deg.tif")

        logging.info("Processed %s", trait)


if __name__ == "__main__":
    main()
