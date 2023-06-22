import pathlib

from utils.datasets import CollectionName


# Configuration settings for 1-Preprocessing.ipynb
class PreprocessingConfig:
    def __init__(self):
        # iNaturalist
        self.iNat_dir = pathlib.Path("./iNaturalist_traits/maps_iNaturalist").absolute()
        self.iNat_name = CollectionName.INAT

        # WorldClim
        self.WC_dir = pathlib.Path("./data/worldclim/bio/").absolute()
        self.WC_bio_ids = ["1", "4", "7", "12", "13", "14", "13-14", "15"]
        self.WC_name = CollectionName.WORLDCLIM

        # MODIS
        self.MODIS_dir = pathlib.Path("./data/modis/").absolute()
        self.MODIS_name = CollectionName.MODIS

        # Soil
        self.soil_dir = pathlib.Path("./data/soil").absolute()
        self.soil_name = CollectionName.SOIL

        self.plot_traits = False

        # Enable to run the GEE exports
        self.gee_export = False
        # Enable to run the resampling operations that write to disk
        self.resamp_to_disk = False
