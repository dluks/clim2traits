from pathlib import Path

import rioxarray as riox

from utils.geodata import da_to_ds, ds_to_netcdf


def main():
    """Convert all existing trait map products into a netCDF dataset, along with the
    new Dong et al., 2023 product."""
    prod_fns = list(
        Path("data/other-products/all-prods_stacks_sla-nit-nita_05D_2022-02-14").glob(
            "*.grd"
        )
    )
    dong_fns = list(Path("data/other-products/dong").glob("*.nc"))

    prod_mapping = {"_nita_": "_Na", "_nit_": "_Nmass", "_sla_": "_SLA"}

    for fn in prod_fns:
        dong_fn = None
        # Find corresponding dong file using prod_mapping
        for key, val in prod_mapping.items():
            if key in fn.name:
                for d_fn in dong_fns:
                    if val in d_fn.name:
                        dong_fn = d_fn
                        break

        if dong_fn is None:
            raise ValueError(f"Could not find dong file for {fn.name}")

        print(f"Processing {fn.name} and {dong_fn.name}")

        all_prods = riox.open_rasterio(
            fn,
            masked=True,
        )

        dong = riox.open_rasterio(dong_fn, masked=True)

        all_prods = all_prods.rio.reproject("epsg:4326")
        all_prods = da_to_ds(all_prods)  # type: ignore

        dong = dong.mean(dim="z")
        dong = dong.rio.write_crs("epsg:4326")
        dong = dong.rio.reproject_match(all_prods)
        dong = dong.assign_coords({"x": all_prods.x, "y": all_prods.y})

        all_prods["Dong"] = dong

        all_prods = all_prods.assign_attrs(
            {"long_name": [str(name) for name in all_prods.data_vars]}
        )

        # Resample to 2 degrees
        all_prods_2deg = all_prods.rio.reproject(
            "epsg:4326", resolution=2, resampling=5
        )

        out_dir_05 = Path(
            fn.parents[1], fn.parent.name.replace("2022-02-14", "2023-12-15")
        )
        out_dir_05.mkdir(exist_ok=True, parents=True)

        out_dir_2 = Path(out_dir_05.parent, out_dir_05.name.replace("05D", "2D"))
        out_dir_2.mkdir(exist_ok=True, parents=True)

        out_fn_05 = out_dir_05 / f"{fn.stem.replace('2022-02-14', '2023-12-15')}.nc"
        out_fn_2 = (
            out_dir_2
            / f"{fn.stem.replace('2022-02-14', '2023-12-15').replace('05D', '2D')}.nc"
        )

        if out_fn_05.exists():
            out_fn_05.unlink()

        if out_fn_2.exists():
            out_fn_2.unlink()

        ds_to_netcdf(all_prods, out_fn_05)
        ds_to_netcdf(all_prods_2deg, out_fn_2)

        print("Wrote", out_fn_05)
        print("Wrote", out_fn_2)


if __name__ == "__main__":
    main()
