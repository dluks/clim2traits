import argparse
import gc
import logging
import multiprocessing
import os
from pathlib import Path
from typing import Sequence, Union

import geopandas as gpd
import xarray as xr
from geocube.api.core import make_geocube
from rioxarray.merge import merge_datasets
from shapely.geometry import Polygon

from utils.geodata import (
    ds_to_gtiff,
    ds_to_netcdf,
    get_trait_name_from_data_name,
    num_to_str,
    pack_ds,
    pad_ds,
)


def get_trait_from_gdf(gdf: gpd.GeoDataFrame, resolution: Union[float, int]) -> str:
    """Get the trait name from a prediction GDF."""
    return gdf.columns[0].replace("_05deg", "") + f"_{num_to_str(resolution)}deg"


def gdf_to_final_ds(fn: Union[str, os.PathLike], resolution: Union[int, float]):
    """Convert a prediction GDF to a final Dataset ready for publication."""
    fn = Path(fn)

    gdf = gpd.read_parquet(fn)
    trait_var = get_trait_from_gdf(gdf, resolution)
    trait_full_name = get_trait_name_from_data_name(trait_var)

    gdf = gdf.rename(columns={"CoV": "COV", gdf.columns[0]: trait_var})

    extent = Polygon.from_bounds(*(-180, -60, 180, 90))

    if fn.stem.startswith("tile"):
        extent = [int(x) for x in fn.stem.split("_")[1:]]
        extent = Polygon.from_bounds(*extent)

    if "AOA" in gdf.columns:
        masked_trait = gdf.columns[4]
    else:
        masked_trait = ""

    ds = make_geocube(
        gdf,
        measurements=gdf.columns.difference(["geometry", masked_trait]).tolist(),
        resolution=(-resolution, resolution),
        output_crs="EPSG:4326",
        geom=extent,
    )

    ds = ds.rio.write_crs("epsg:4326")
    ds = ds.assign_attrs({"crs": ds.rio.crs.to_string()})
    ds = pad_ds(ds)

    for dv in ds.data_vars:
        if str(dv) == "AOA":
            ds[dv] = ds[dv].fillna(0)
            ds[dv] = ds[dv].astype("int16")
            ds[dv].attrs["long_name"] = "Area of Applicability"
        if str(dv) == "DI":
            ds[dv] = ds[dv].astype("float32")
            ds[dv].attrs["long_name"] = "Dissimilarity Index"
        if str(dv) == "COV":
            ds[dv] = ds[dv].astype("float32")
            ds[dv].attrs["long_name"] = "Coefficient of Variation"
        if "TRYgapfilled" in str(dv):
            ds[dv] = ds[dv].astype("float32")
            ds[dv].attrs["long_name"] = trait_full_name

    ds = pack_ds(ds)

    log = " (log-transformed)" if "_ln" in trait_var else ""

    ds = ds.assign_attrs(
        creator_name="Daniel Lusk",
        contact="lusk@posteo.net",
        trait=trait_var,
        trait_description=f"{trait_full_name}{log}",
        version="0.1",
    )

    logging.info("Converted %s", fn.name)

    return ds


def pred_to_ds(
    filenames: Sequence[Union[str, os.PathLike]],
    resolution: Union[int, float],
    tiled: bool = False,
    num_procs: int = 1,
) -> xr.Dataset:
    """Convert prediction GDF(s) into a single Dataset."""
    if tiled:
        predictor_set = Path(filenames[0]).parents[4].name.split("tiled_5x5_deg_")[1]

        if num_procs > 1:
            with multiprocessing.Pool(num_procs) as pool:
                datasets = pool.starmap(
                    gdf_to_final_ds, [(fn, resolution) for fn in filenames]
                )
        else:
            datasets = [gdf_to_final_ds(fn, resolution) for fn in filenames]

        logging.info("Merging datasets...")
        ds = merge_datasets(datasets)
    else:
        predictor_set = Path(filenames[0]).stem.split("_predict")[0]
        ds = gdf_to_final_ds(filenames[0], resolution)

    ds = ds.assign_attrs(
        predictor_set=predictor_set,
    )

    return ds


def write_pred_ds(ds, out_fn: Union[str, os.PathLike], dry_run: bool = False):
    """Write a prediction Dataset."""
    out_fn = Path(out_fn)

    if dry_run:
        logging.info("Wrote %s (DRY-RUN)", out_fn)
        return

    if out_fn.suffix == ".nc":
        ds_to_netcdf(ds, out_fn)
    elif out_fn.suffix == ".tif":
        ds_to_gtiff(ds, out_fn)
    else:
        raise ValueError(f"Unknown file extension: {out_fn.suffix}")
    logging.info("Wrote %s", out_fn)
    return


def bounds_to_str(bounds: tuple) -> str:
    """Convert a tuple of bounds to a string."""
    return f"{'_'.join([str(x) for x in bounds])}"


def pred_to_ds_and_write(
    filename: Union[str, os.PathLike],
    resolution: Union[int, float],
    out_dir: Union[str, os.PathLike],
    tiled: bool = False,
    dry_run: bool = False,
    num_procs: int = 1,
    overwrite: bool = False,
):
    """Writes a prediction GDF to a NetCDF file."""

    if tiled:
        tile_fns = sorted(list(Path(filename).glob("*.parq")))

        trait_var = get_trait_from_gdf(gpd.read_parquet(tile_fns[0]), resolution)

        if not overwrite:
            out_fn = Path(out_dir, f"{trait_var}.nc")
            if out_fn.exists():
                logging.info("%s already exists. Skipping...", out_fn)
                return

        ds = pred_to_ds(tile_fns, resolution, tiled, num_procs)
        # logging.info("Writing %s...", out_fn)
        # write_pred_ds(ds, out_fn, dry_run=dry_run)

    else:
        ds = pred_to_ds([filename], resolution)

    out_fn = Path(out_dir, f"{ds.attrs['trait']}.nc")

    if not overwrite:
        if out_fn.exists():
            logging.info("%s already exists. Skipping...", out_fn)
            return

    logging.info("Writing %s...", out_fn)

    write_pred_ds(ds, out_fn, dry_run=dry_run)

    ds.close()

    del ds
    gc.collect()


def main():
    """Write trait maps to NetCDF."""
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, help="Path to dataset predictions.")
    parser.add_argument(
        "-r",
        "--resolution",
        type=float,
        default=0.5,
        help="Resolution of prediction set in degrees.",
    )
    parser.add_argument(
        "-p",
        "--pft",
        type=str,
        default="Shrub_Tree_Grass",
        help="Plant functional type.",
    )
    parser.add_argument(
        "-t",
        "--tiled",
        action="store_true",
        help="Tiled predictions (Should usually be combined with --num-procs > 1 if "
        "there are more than a few tiles).",
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="Overwrite files if they already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run.")
    parser.add_argument(
        "-n", "--num-procs", type=int, default=1, help="Number of processes."
    )
    args = parser.parse_args()

    dataset = args.dataset
    fns = (
        Path(dataset).glob("*/*/tiled_5x5_deg")
        if args.tiled
        else Path(dataset).glob("*/*/*.parq")
    )
    parent_dir = Path(
        "maps", args.pft, "05deg_models", f"{num_to_str(args.resolution)}deg"
    )

    if args.verbose or args.dry_run:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        )

        logging.info(
            f"Processing {dataset}"
            if not args.dry_run
            else f"Processing {dataset} (DRY-RUN)"
        )

    if not args.dry_run:
        parent_dir.mkdir(parents=True, exist_ok=True)

    if args.num_procs == -1:
        args.num_procs = multiprocessing.cpu_count()

    if args.num_procs > 1 and not args.tiled:
        with multiprocessing.Pool(args.num_procs) as pool:
            pool.starmap(
                pred_to_ds_and_write,
                [
                    (
                        fn,
                        args.resolution,
                        parent_dir,
                        args.tiled,
                        args.dry_run,
                        1,
                        args.overwrite,
                    )
                    for fn in fns
                ],
            )
    else:
        for fn in fns:
            pred_to_ds_and_write(
                fn,
                args.resolution,
                parent_dir,
                args.tiled,
                args.dry_run,
                args.num_procs,
                args.overwrite,
            )

    print("Done.")


if __name__ == "__main__":
    main()
