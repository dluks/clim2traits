from functools import reduce

import ee

from utils import gee

# ee.Authenticate()  # Uncomment if not already authenticated
ee.Initialize()

# Get MODIS Terra Surface Reflectance image collection for its first five years of operation
# (2000-03-01 - 2001-03-01)
ds, de = "2000-03-01", "2020-03-02"
bands = [
    "sur_refl_b01",
    "sur_refl_b02",
    "sur_refl_b03",
    "sur_refl_b04",
    "sur_refl_b05",
]
modis_tsr = ee.ImageCollection("MODIS/061/MOD09GA").filterDate(ds, de)

# Mask clouds
qa_band = "state_1km"
modis_tsr_masked = gee.mask_clouds(modis_tsr, qa_band)

# Aggregate the image collection into monthly averages for each band
tsr_bands_monthly = []
for band in bands:
    monthly = gee.aggregate_ic(modis_tsr_masked.select(band), ds, de)
    tsr_bands_monthly.append(monthly)

tsr_monthly_means = []
for band_ic in tsr_bands_monthly:
    tsr_monthly_means.append(gee.aggregate_ic_monthly(band_ic))

# 1. Combine the five bands into a single image collection
# 2. Reproject the image collection to EPSG:4326 with a scale of 1km
# 3. Unmask the image collection and convert to int16 (because NoData values are
#    replaced with 0 when converting to int16)

final_ic = (
    # tsr_monthly_means[0]
    # .combine(tsr_monthly_means[1])
    # .combine(tsr_monthly_means[2])
    # .combine(tsr_monthly_means[3])
    # .combine(tsr_monthly_means[4])
    # ee.List(tsr_monthly_means).iterate(lambda x, y: x.combine(y), tsr_monthly_means[0])
    # # This may be the best approach. Casts the list to a FeatureCollection to an IC
    # ee.ImageCollection(ee.FeatureCollection(tsr_monthly_means).flatten())
    reduce(lambda x, y: x.combine(y), tsr_monthly_means)  # this probably won't work
    .map(lambda image: image.reproject(crs="EPSG:4326", crsTransform=None, scale=1000))
    .map(lambda image: image.unmask(-32768).toInt16())
)

# # Reproject the image collection to EPSG:4326 with a scale of 1km
# tsr_monthly_means = tsr_monthly_means.map(
#     lambda x: x.reproject(crs="EPSG:4326", crsTransform=None, scale=1000)
# )

# # Unmask the image collection and convert to int16 (because NoData values are replaced
# # with 0 when converting to int16)
# tsr_monthly_means = tsr_monthly_means.map(lambda x: x.unmask(-32768))
# tsr_monthly_means = tsr_monthly_means.map(lambda x: x.toInt16())

# Export images to Google Drive
gee.export_collection(collection=final_ic, folder="multiband_monthly_averages")
