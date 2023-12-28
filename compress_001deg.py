import gc
from pathlib import Path

import rioxarray as riox
from tqdm import tqdm

pfts = list(Path("maps").glob("*"))
pfts = [pft for pft in pfts if pft.is_dir()]

fns = [list(pft.glob("*/001deg/*.tif")) for pft in pfts]

# Flatten fns
fns = [fn for sublist in fns for fn in sublist]


for fn in tqdm(sorted(list(fns))):
    r = riox.open_rasterio(fn, masked=True)

    # append "_compressed" to the filename
    out_fn = fn.parent / f"{fn.stem}_compressed.tif"

    # Rewrite the raster with compression
    r.rio.to_raster(
        out_fn,
        compress="zstd",
        tiled=True,
        predictor=2,
        num_threads=20,
        tags={
            "compression": "zstd",
            "compression_predictor": 2,
            "resolution": 0.01,
            "resolution_unit": "degree",
        }
        # windowed=True,
    )

    r.close()

    del r
    gc.collect()
