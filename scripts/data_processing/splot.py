import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as riox

from utils.geodata import num_to_str, rasterize_points, write_raster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

### CONFIG
RANGE_05 = True
PFTs = ["Shrub_Tree_Grass", "Shrub_Tree", "Grass"]
RES = 0.01
BBOX = (-180, -60, 180, 90)


def main():
    """Convert sPlot data to raster grids."""
    for PFT in PFTs:
        splot_cwm = pd.read_csv(
            f"data/gbif-splot_raw/sPlot_cwms/sPlotOpen_TRYgapfilled_cwm{PFT}.csv"
        ).drop(columns=["PlotObservationID", "Releve_area"])

        parent_out = Path(f"GBIF_trait_maps/global_maps/{PFT}/{num_to_str(RES)}deg")
        parent_out.mkdir(parents=True, exist_ok=True)

        traits = splot_cwm.columns.difference(["Latitude", "Longitude"])

        for trait in traits:
            points = (
                splot_cwm[["Longitude", "Latitude", trait]]
                .dropna(subset=[trait])
                .values
            )

            if RANGE_05:
                out_dir = parent_out / "05_range"
                out_dir.mkdir(parents=True, exist_ok=True)

                splot_05 = riox.open_rasterio(
                    Path(
                        f"GBIF_trait_maps/global_maps/{PFT}/05deg/",
                        f"sPlot_TRYgapfilled_{trait}_05deg.grd",
                    ),
                    masked=True,
                )

                if isinstance(splot_05, list):
                    raise ValueError(
                        "Input file is a list. Please provide a single file."
                    )

                # Get min and max values of splot_05
                min_val = splot_05.min().values.item()
                max_val = splot_05.max().values.item()

                # Drop points outside min_val and max_val, and store count of number of
                # points dropped
                before = points.shape[0]
                points = points[
                    np.logical_and(points[:, 2] >= min_val, points[:, 2] <= max_val)
                ]
                after = points.shape[0]
                logging.info("Dropped %d points outside range", before - after)
            else:
                out_dir = parent_out / "orig"
                out_dir.mkdir(parents=True, exist_ok=True)

            grid = rasterize_points(points, RES, BBOX)

            out_fn = out_dir / f"sPlot_TRYgapfilled_{trait}_{num_to_str(RES)}deg.tif"
            write_raster(
                grid,
                RES,
                BBOX,
                out_fn,
            )

            logging.info("Wrote %s", out_fn)


if __name__ == "__main__":
    main()
