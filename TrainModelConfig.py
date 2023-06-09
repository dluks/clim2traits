from utils.data_retrieval import get_fns


# Configuration settings for 2-TrainModel.ipynb
class TrainModelConfig:
    def __init__(self):
        # Set to None if resolution should be separate for each dataset
        MAIN_RES = "0.5_deg"

        # iNaturalist
        self.iNat_dir = "./iNaturalist_traits/maps_iNaturalist"
        self.iNat_res = MAIN_RES or "0.5_deg"
        self.iNat_transform = "ln"
        self.iNat_fns = get_fns(
            self.iNat_dir, self.iNat_res, "iNat", self.iNat_transform
        )

        # WorldClim
        self.WC_dir = "./data/worldclim/bio/"
        self.WC_res = MAIN_RES or "0.5_deg"
        self.WC_bios = [1, 4, 7, 12, 13, 14, 15]
        self.WC_fns = get_fns(self.WC_dir, self.WC_res, "wc", bios=self.WC_bios)

        # MODIS
        self.MODIS_dir = "./data/modis/"
        self.MODIS_res = MAIN_RES or "0.5_deg"
        self.MODIS_fns = get_fns(self.MODIS_dir, self.MODIS_res, "modis")

        # Soil
        self.soil_dir = "./data/soil"
        self.soil_res = MAIN_RES or "0.5_deg"
        self.soil_fns = get_fns(self.soil_dir, self.soil_res, "soil")

        self.SAVE_AUTOCORRELATION_RANGES = False

        self.RNG_STATE = 100919
