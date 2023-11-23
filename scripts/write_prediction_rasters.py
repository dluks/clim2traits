import argparse
import json
import logging
import multiprocessing
import os
import re
from pathlib import Path
from typing import Union

import geopandas as gpd
import xarray as xr
from geocube.api.core import make_geocube

from utils.geodata import ds_to_netcdf, num_to_str


def get_trait_id_from_data_name(data_name: str) -> str:
    """Get trait id from data name, e.g. GBIF_TRYgapfilled_X1080_05deg_mean_ln -> 1080"""
    trait_id = re.search(r"X\d+", data_name).group()
    trait_id = trait_id.replace("X", "")
    return trait_id


def get_trait_name_from_trait_id(trait_id: str) -> str:
    """Get trait name from trait id, e.g. 1080 -> Root length per root dry mass
    (specific root length, SRL) (log-transformed)"""
    with open("./trait_id_to_trait_full_name.json", encoding="utf-8") as f:
        mapping = json.load(f)
        trait_name = mapping[trait_id]
    return trait_name


def get_trait_name_from_data_name(data_name: str) -> str:
    """Get trait name from data name, e.g. GBIF_TRYgapfilled_X1080_05deg_mean_ln ->
    Root length per root dry mass (specific root length, SRL) (log-transformed)"""
    trait_id = get_trait_id_from_data_name(data_name)
    trait_name = get_trait_name_from_trait_id(trait_id)
    return trait_name


def pred_to_ds(
    filename: Union[str, os.PathLike],
    resolution: Union[int, float],
) -> xr.Dataset:
    """Write a prediction set to a NetCDF file."""
    pred = gpd.read_parquet(filename)
    pred = pred.rename(columns={"AOA": "AOA", "CoV": "COV"})
    predictor_set = Path(filename).stem.split("_predict")[0]
    trait_var = pred.columns[0].replace("_05deg", "") + f"_{num_to_str(resolution)}deg"
    trait_full_name = get_trait_name_from_data_name(trait_var)
    masked_trait = pred.columns[4]

    ds = make_geocube(
        vector_data=pred,
        measurements=pred.columns.difference(["geometry", masked_trait]).tolist(),
        resolution=(resolution, -resolution),
        output_crs="EPSG:4326",
    )

    log = " (log-transformed)" if "_ln" in trait_var else ""

    data_vars_description = (
        "AOA: Area of Applicability; threshold = 0.95 of DI (Meyer and Pebesma, 2021)\n"
        "COV: Coefficient of Variation\n"
        "DI: Dissimilarity Index (Meyer and Pebesma, 2021)\n"
        f"{trait_var}: Extrapolated trait value{log}"
    )

    ds = ds.assign_attrs(
        creator_name="Daniel Lusk",
        contact="lusk@posteo.net",
        trait=trait_var,
        trait_description=f"{trait_full_name}{log}",
        data_vars_description=data_vars_description,
        predictor_set=predictor_set,
        version="0.1",
    )

    return ds


def write_pred_ds(ds, out_dir: Union[str, os.PathLike], dry_run: bool = False):
    """Write a prediction Dataset to a NetCDF file."""
    out_fn = Path(out_dir, f"{ds.attrs['trait']}.nc")

    if dry_run:
        logging.info("Wrote %s (DRY-RUN)", out_fn)
        return

    ds_to_netcdf(ds, out_fn)
    logging.info("Wrote %s", out_fn)
    return


def write_pred(
    filename: Union[str, os.PathLike],
    resolution: Union[int, float],
    out_dir: Union[str, os.PathLike],
    dry_run: bool = False,
):
    """Writes a prediction GDF to a NetCDF file."""
    ds = pred_to_ds(filename, resolution)
    write_pred_ds(ds, out_dir, dry_run=dry_run)


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
        "--pft", type=str, default="Shrub_Tree_Grass", help="Plant functional type."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print verbose output.",
    )
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run.")
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes.")
    args = parser.parse_args()

    dataset = args.dataset
    fns = Path(dataset).glob("*/*/*.parq")
    parent_dir = Path(f"maps/{args.pft}/05deg_models/{num_to_str(args.resolution)}deg")

    if args.verbose or args.dry_run:
        logging.basicConfig(level=logging.INFO)

        logging.info(
            f"Processing {dataset}"
            if not args.dry_run
            else f"Processing {dataset} (DRY-RUN)"
        )

    if not args.dry_run:
        parent_dir.mkdir(parents=True, exist_ok=True)

    if args.num_procs != 1:
        if args.num_procs == -1:
            args.num_procs = multiprocessing.cpu_count()
        with multiprocessing.Pool(args.num_procs) as pool:
            pool.starmap(
                write_pred,
                [(fn, args.resolution, parent_dir, args.dry_run) for fn in fns],
            )
    else:
        for fn in fns:
            write_pred(fn, args.resolution, parent_dir, dry_run=args.dry_run)

    print("Done.")


if __name__ == "__main__":
    main()
