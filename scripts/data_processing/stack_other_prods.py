#!/usr/bin/env python3
import argparse
from pathlib import Path

from utils.geodata import da_to_ds, ds_to_netcdf, open_raster


def main():
    """Convert all existing trait map products into a netCDF dataset, along with the
    new Dong et al., 2023 product."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dry-run", action="store_true", help="Dry run")
    args = parser.parse_args()

    prod_fns = list(
        Path("data/other-products/all-prods_stacks_sla-nit-nita_05D_2022-02-14").glob(
            "*.grd"
        )
    )
    dong_fns = list(Path("data/other-products/dong").glob("*.nc"))
    wolf_fns_2 = list(
        Path("GBIF_trait_maps/global_maps/Shrub_Tree_Grass/2deg").glob("GBIF*.grd")
    )
    wolf_fns_05 = list(
        Path("GBIF_trait_maps/global_maps/Shrub_Tree_Grass/05deg").glob("GBIF*.grd")
    )
    wolf_fns_02 = list(
        Path("GBIF_trait_maps/global_maps/Shrub_Tree_Grass/02deg").glob("GBIF*.grd")
    )

    product_mapping = {
        "_nita_": ["_Na", "50"],
        "_nit_": ["_Nmass", "14"],
        "_sla_": ["_SLA", "11"],
    }

    for fn in prod_fns:
        dong_fn = None
        wolf_2_fn = None
        wolf_05_fn = None
        wolf_02_fn = None
        # Find corresponding Dong et al. file using prod_mapping
        for key, val in product_mapping.items():
            if key in fn.name:
                dong_fn = [d_fn for d_fn in dong_fns if val[0] in d_fn.name][0]
                wolf_2_fn = [
                    w_fn for w_fn in wolf_fns_2 if f"_X{val[1]}_" in w_fn.name
                ][0]
                wolf_05_fn = [
                    w_fn for w_fn in wolf_fns_05 if f"_X{val[1]}_" in w_fn.name
                ][0]
                wolf_02_fn = [
                    w_fn for w_fn in wolf_fns_02 if f"_X{val[1]}_" in w_fn.name
                ][0]

        if dong_fn is None:
            raise ValueError(f"Could not find Dong file for {fn.name}")

        if wolf_2_fn is None or wolf_05_fn is None or wolf_02_fn is None:
            raise ValueError(f"Could not find Wolf file for {fn.name}")

        # Load the rasters
        all_prods_05 = open_raster(fn, masked=True)
        dong_05 = open_raster(dong_fn, masked=True)
        wolf_2 = open_raster(wolf_2_fn, masked=True).sel(band=2)
        wolf_05 = open_raster(wolf_05_fn, masked=True).sel(band=2)
        wolf_02 = open_raster(wolf_02_fn, masked=True).sel(band=2)

        all_prods_05 = all_prods_05.rio.reproject("epsg:4326")
        all_prods_05 = da_to_ds(all_prods_05)  # type: ignore

        dong_05 = dong_05.mean(dim="z")
        dong_05 = dong_05.rio.write_crs("epsg:4326")
        dong_05 = dong_05.rio.reproject_match(all_prods_05)
        dong_05 = dong_05.assign_coords({"x": all_prods_05.x, "y": all_prods_05.y})

        wolf_05 = wolf_05.rio.reproject_match(all_prods_05)
        wolf_05 = wolf_05.assign_coords({"x": all_prods_05.x, "y": all_prods_05.y})

        all_prods_05 = all_prods_05.assign_attrs(
            {"long_name": [str(name) for name in all_prods_05.data_vars]}
        )

        all_prods_05["Dong"] = dong_05
        all_prods_05.attrs["long_name"].append("Dong")

        # Resample to 2 degrees
        all_prods_2 = all_prods_05.rio.reproject(
            "epsg:4326", resolution=2, resampling=5
        )

        # Only add Wolf 0.5deg after resampling all_prods to 2 deg since we already have
        # native 2 deg Wolf data
        all_prods_05["Wolf"] = wolf_05
        all_prods_05.attrs["long_name"].append("Wolf")

        wolf_2 = wolf_2.rio.reproject("epsg:4326")
        wolf_2 = wolf_2.rio.reproject_match(all_prods_2)
        wolf_2 = wolf_2.assign_coords({"x": all_prods_2.x, "y": all_prods_2.y})

        all_prods_2["Wolf"] = wolf_2
        all_prods_2.attrs["long_name"].append("Wolf")

        # Create the 0.2 degree dataset, starting with Wolf
        all_prods_02 = wolf_02.rio.reproject("epsg:4326")
        all_prods_02.rio.write_crs("epsg:4326", inplace=True)
        all_prods_02 = da_to_ds(all_prods_02, name="Wolf")

        # Configure output files
        out_dir_05 = Path(
            fn.parents[1], fn.parent.name.replace("2022-02-14", "2023-12-15")
        )
        out_dir_05.mkdir(exist_ok=True, parents=True)

        out_dir_2 = Path(out_dir_05.parent, out_dir_05.name.replace("05D", "2D"))
        out_dir_2.mkdir(exist_ok=True, parents=True)

        out_dir_02 = Path(out_dir_05.parent, out_dir_05.name.replace("05D", "02D"))
        out_dir_02.mkdir(exist_ok=True, parents=True)

        out_fn_05 = out_dir_05 / f"{fn.stem.replace('2022-02-14', '2023-12-15')}.nc"
        out_fn_2 = (
            out_dir_2
            / f"{fn.stem.replace('2022-02-14', '2023-12-15').replace('05D', '2D')}.nc"
        )
        out_fn_02 = (
            out_dir_02
            / f"{fn.stem.replace('2022-02-14', '2023-12-15').replace('05D', '02D')}.nc"
        )

        if out_fn_05.exists():
            out_fn_05.unlink()

        if out_fn_2.exists():
            out_fn_2.unlink()

        if out_fn_02.exists():
            out_fn_02.unlink()

        if not args.dry_run:
            ds_to_netcdf(all_prods_05, out_fn_05)
            ds_to_netcdf(all_prods_2, out_fn_2)
            ds_to_netcdf(all_prods_02, out_fn_02)

        print("Wrote", out_fn_05)
        print("Wrote", out_fn_2)
        print("Wrote", out_fn_02)


if __name__ == "__main__":
    main()
