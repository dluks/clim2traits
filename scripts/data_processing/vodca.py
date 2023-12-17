import argparse
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr

from utils.dataset_tools import Unit
from utils.datasets import CollectionName, Dataset, resample_dataset


def daily_to_multiyear_monthly(fpaths: list[Path], out_dir: Path) -> None:
    """Converts daily data to multiyear monthly data and saves to disk"""
    ds = xr.open_mfdataset(
        fpaths,
        engine="h5netcdf",
        chunks=360,
    )

    band_name = fpaths[0].parent.parent.name.split("-")[0]

    years = np.sort(np.unique([fpath.parent.name for fpath in fpaths]))
    start, end = years[0], years[-1]

    ds_name = f"{band_name}_{start}_{end}_multiyear_mean"

    ds = ds.drop_vars(["sensor_flag", "processing_flag"])
    ds = ds.rename({"vod": band_name})
    ds = ds.rename({"lon": "x", "lat": "y"})

    ds_05 = (
        ds.coarsen(x=2, y=2, boundary="exact")
        .mean()  # type: ignore (typing bug in xarray)
        .resample(time="1MS")
        .mean()
        .groupby("time.month")
        .mean("time")
    )
    ds_05.attrs["geospatiallatresolution"] = "0.5 degree"
    ds_05.attrs["geospatiallonresolution"] = "0.5 degree"

    ds_025 = ds.resample({"time": "1MS"}).mean().groupby("time.month").mean("time")
    ds_025.attrs["geospatiallatresolution"] = "0.25 degree"
    ds_025.attrs["geospatiallonresolution"] = "0.25 degree"

    for i in range(1, 13):
        month_05 = ds_05.sel(month=i)[f"{band_name}"]
        month_05.rio.write_crs("EPSG:4326", inplace=True)
        month_05.rio.write_nodata(np.nan, encoded=True, inplace=True)
        month_05.attrs["long_name"] = f"Vegetation optical depth {band_name} band"

        month_025 = ds_025.sel(month=i)[f"{band_name}"]
        month_025.rio.write_crs("EPSG:4326", inplace=True)
        month_025.rio.write_nodata(np.nan, encoded=True, inplace=True)
        month_025.attrs["long_name"] = f"Vegetation optical depth {band_name} band"

        out_name = Path(f"{ds_name}_m{i:02d}.tif")

        out_dir_05 = out_dir / "0.5_deg" / "new"
        out_dir_05.mkdir(parents=True, exist_ok=True)
        out_dir_025 = out_dir / "0.25_deg" / "new"
        out_dir_025.mkdir(parents=True, exist_ok=True)

        month_05.rio.to_raster(
            out_dir_05 / out_name,
            dtype="float32",
            compress="zstd",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            predictor=2,
            num_threads=20,
            compute=False,
        )

        month_025.rio.to_raster(
            out_dir_025 / out_name,
            dtype="float32",
            compress="zstd",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            predictor=2,
            num_threads=20,
            compute=False,
        )

        print(f"Exported {out_name}")


def main():
    """Main function for converting VODCA data to multiyear monthly data"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--standardize", action="store_true")
    parser.add_argument("--fill-holes", action="store_true")
    parser.add_argument("--resample_05", action="store_true")
    parser.add_argument("--resample_001", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    out_dir = Path("./data/vodca")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.standardize:
        bands = ["Ku-band", "X-band"]
        band_paths = [
            list(Path(f"./data/vodca/source/{band}").glob("*/*.nc")) for band in bands
        ]

        cpus = len(bands)
        with Pool(cpus) as p:
            p.starmap(daily_to_multiyear_monthly, zip(band_paths, [out_dir] * cpus))

    if args.fill_holes:
        vodca = Dataset(
            res=0.25,
            unit=Unit.DEGREE,
            collection_name=CollectionName.VODCA,
        )

        vodca.fill_holes(method="linear")

    if args.resample_05:
        vodca = Dataset(
            res=0.25,
            unit=Unit.DEGREE,
            collection_name=CollectionName.VODCA,
        )

        resample_dataset(
            vodca, 0.5, unit=Unit.DEGREE, resample_alg=5, dry_run=args.dry_run
        )

    if args.resample_001:
        vodca = Dataset(
            res=0.25,
            unit=Unit.DEGREE,
            collection_name=CollectionName.VODCA,
        )

        resample_dataset(
            vodca, 0.01, unit=Unit.DEGREE, resample_alg=1, dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()
