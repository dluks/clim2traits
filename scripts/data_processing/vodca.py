from multiprocessing import Pool
from pathlib import Path

import numpy as np
import xarray as xr


def daily_to_multiyear_monthly(fpaths: list[Path], out_dir: Path) -> None:
    ds = xr.open_mfdataset(
        fpaths,
        engine="h5netcdf",
        chunks=360,
    )

    band_name = fpaths[0].parent.parent.name.split("-")[0]

    years = np.sort(np.unique([fpath.parent.name for fpath in fpaths]))
    start, end = years[0], years[-1]

    ds_name = f"{band_name}_{start}_{end}_multiyear_mean"

    """Converts daily data to multiyear monthly data and saves to disk"""
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

    ds_025 = ds.resample(time="1MS").mean().groupby("time.month").mean("time")
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
            num_threads=18,
        )

        month_025.rio.to_raster(
            out_dir_025 / out_name,
            dtype="float32",
            compress="zstd",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            predictor=2,
            num_threads=18,
        )

        print(f"Exported {out_name}")


if __name__ == "__main__":
    out_dir = Path("./data/vodca")
    out_dir.mkdir(parents=True, exist_ok=True)

    bands = ["C-Band", "Ku-band", "X-band"]
    band_paths = [
        list(Path(f"./data/vodca/source/{band}").glob("20*/*.nc")) for band in bands
    ]

    with Pool(3) as p:
        p.starmap(daily_to_multiyear_monthly, zip(band_paths, [out_dir] * 3))
