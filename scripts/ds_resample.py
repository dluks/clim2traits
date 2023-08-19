from utils.dataset_tools import Unit
from utils.datasets import CollectionName, Dataset, resample_dataset

vodca = Dataset(
    res=0.25,
    unit=Unit.DEGREE,
    collection_name=CollectionName.VODCA,
)

soil = Dataset(
    res=1,
    unit=Unit.KILOMETER,
    collection_name=CollectionName.SOIL,
)


if __name__ == "__main__":
    # Uncomment to resample the VODCA dataset to another resolution
    resample_dataset(dataset=soil, resolution=0.01, unit=Unit.DEGREE, resample_alg=4)
