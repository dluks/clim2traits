import argparse

from utils.dataset_tools import Unit
from utils.datasets import CollectionName, Dataset, resample_dataset

modis = Dataset(
    res=1,
    unit=Unit.KILOMETER,
    collection_name=CollectionName.MODIS,
)

vodca = Dataset(
    res=0.25,
    unit=Unit.DEGREE,
    collection_name=CollectionName.VODCA,
)

soil = Dataset(
    res=0.01,
    unit=Unit.DEGREE,
    collection_name=CollectionName.SOIL,
)

wc = Dataset(
    res=30,
    unit=Unit.SECOND,
    collection_name=CollectionName.WORLDCLIM,
)


if __name__ == "__main__":
    # Uncomment to resample the VODCA dataset to another resolution
    # datasets = [modis, wc, vodca, soil]
    datasets = [soil]

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res",
        type=float,
        default=0.5,
        help="Resolution in degrees to resample dataset to",
    )
    parser.add_argument("--num-procs", type=int, default=1, help="Number of processes")
    parser.add_argument("--dry-run", action="store_true", help="Dry run")
    parser.add_argument(
        "-o", "--overwrite", action="store_true", help="Overwrite files if they exist."
    )

    args = parser.parse_args()

    for dataset in datasets:
        resample_dataset(
            dataset,
            args.res,
            Unit.DEGREE,
            dry_run=args.dry_run,
            num_procs=args.num_procs,
            overwrite=args.overwrite,
        )
