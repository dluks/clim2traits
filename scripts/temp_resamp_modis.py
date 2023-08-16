from pathlib import Path

from utils.gdal import resample_gdal

fns = Path("./data/modis/1_km_old").glob("*.tif")
RES = 0.00898315284120171538

for fn in fns:
    out_dir = Path("./data/modis/1_km")
    out_dir.mkdir(exist_ok=True, parents=True)
    out_fn = out_dir / fn.name
    resample_gdal(in_fn=str(fn), out_fn=str(out_fn), res=RES, globe=True)
