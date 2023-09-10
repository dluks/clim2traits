import numpy as np
from spacv.visualisation import plot_autocorrelation_ranges

from TrainModelConfig import TrainModelConfig
from utils.dataset_tools import FileExt, Unit
from utils.datasets import CollectionName, DataCollection, Dataset, GBIFBand

config = TrainModelConfig()

wc = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    collection_name=CollectionName.WORLDCLIM,
)

modis = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    collection_name=CollectionName.MODIS,
)

soil = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    collection_name=CollectionName.SOIL,
)

vodca = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    collection_name=CollectionName.VODCA,
)

# X = DataCollection([wc, modis, soil, vodca])

X = DataCollection.from_collection(
    "data/collections/MOD09GA.061_ISRIC_soil_WC_BIO_VODCA_0.5_deg_nan-strat=all.parquet"
)

gbif = Dataset(
    res=0.5,
    unit=Unit.DEGREE,
    collection_name=CollectionName.GBIF,
    band=GBIFBand.MEAN,
    file_ext=FileExt.GRID,
)

y = DataCollection([gbif])

# Clip data to between 80N and 60S
y.df = y.df.cx[:, -60:80]

coords = y.df["geometry"]
data = y.df[y.cols]

print(y.df.geometry.y.min())
print(y.df.geometry.y.max())
_, _, ranges = plot_autocorrelation_ranges(
    coords,
    data,
    config.LAGS,
    config.BW,
    distance_metric="haversine",
    verbose=True,
    workers=10,
)

np.save("ranges_y_30000.npy", np.asarray(ranges))
