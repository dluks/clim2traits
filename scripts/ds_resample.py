import os

from utils.dataset_tools import Unit
from utils.datasets import CollectionName, Dataset, resample_dataset

vodca = Dataset(
    res=0.25,
    unit=Unit.DEGREE,
    collection_name=CollectionName.VODCA,
)


if __name__ == "__main__":
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    # Uncomment to resample the VODCA dataset to another resolution
    resample_dataset(
        dataset=vodca,
        resolution=0.01,
        unit=Unit.DEGREE,
        format="netcdf",
        mf=False,
    )
