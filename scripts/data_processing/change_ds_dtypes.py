import gc
import multiprocessing
from pathlib import Path

import rioxarray as riox

from utils.geodata import ds_to_netcdf, pack_ds, pad_ds


def pack_001_ds(fn: Path):
    """Pack a 001deg sPlot map."""
    ds = riox.open_rasterio(fn, masked=False)
    ds = ds.to_dataset(dim="band")
    attrs_to_del = [
        "scale_factor",
        "add_offset",
        "_FillValue",
        "compression",
        "compression_predictor",
        "band_descriptions",
        "resolution",
        "resolution_unit",
        "name",
    ]
    for attr in attrs_to_del:
        del ds.attrs[attr]
    ds = ds.rio.write_crs("epsg:4326")
    ds = ds.assign_attrs({"crs": ds.rio.crs.to_string()})

    ds = pad_ds(ds)

    for i, dv in enumerate(ds.data_vars):
        ds = ds.rename_vars({dv: ds.attrs["long_name"][i]})
    del ds.attrs["long_name"]

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
            ds[dv].attrs["long_name"] = ds.attrs["trait_description"]

    ds = pack_ds(ds)

    out_fn = fn.parent / "packed" / f"{fn.stem}.nc"
    out_fn.parent.mkdir(parents=True, exist_ok=True)

    ds_to_netcdf(ds, out_fn)
    ds.close()
    del ds
    gc.collect()
    print(f"Wrote {out_fn}")


def main():
    """Pack all 001deg sPlot maps."""
    pft_dirs = Path("maps").glob("*")
    pft_dirs = [pft_dir for pft_dir in pft_dirs if pft_dir.is_dir()]

    fns = [pft_dir.glob("05deg_models/001deg/*.tif") for pft_dir in pft_dirs]
    # flatten list of lists
    fns = sorted([fn for sublist in fns for fn in sublist])
    packed_fns = [fn.parent / "packed" / f"{fn.stem}.nc" for fn in fns]
    packed_fns = [fn for fn in packed_fns if fn.exists()]
    print("all fns:", len(fns))
    print("packed fns:", len(packed_fns))

    # find files that have not been packed yet
    fns_to_pack = []
    for fn in fns:
        packed_fn = Path(fn.parent / "packed" / f"{fn.stem}.nc")
        # get the size of packed_fn. if it exceeds 400 MB, remove it from fns
        if packed_fn.exists():
            if fn.stem.endswith("AOA"):
                # remove if size > 500 MB
                if packed_fn.stat().st_size / 1024 / 1024 < 500:
                    fns_to_pack.append(fn)
            else:
                if packed_fn.stat().st_size / 1024 / 1024 < 400:
                    fns_to_pack.append(fn)
        else:
            fns_to_pack.append(fn)

    print("fns to pack:", len(fns_to_pack))

    with multiprocessing.Pool(2) as pool:
        pool.map(pack_001_ds, fns_to_pack)

    print("Done.")


if __name__ == "__main__":
    main()
