import os

import numpy as np

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

        # Spatial CV
        if os.path.exists("./ranges.npy"):
            self.AUTOCORRELATION_RANGE = np.median(np.load("ranges.npy"))
        else:
            raise ValueError("Cannot locate autocorrelation ranges")

        self.DEGREE = 111325  # Standard value for 1 degree in meters at the equator
        UPPER_BOUND = np.round(self.DEGREE * 120)
        STEP = np.round(UPPER_BOUND / 30)
        self.BW = STEP // 2
        self.LAGS = np.arange(0, UPPER_BOUND, STEP)

        # TRAINING
        self.RESULTS_DIR = os.path.abspath("./results")
        if not os.path.exists(self.RESULTS_DIR):
            os.makedirs(self.RESULTS_DIR)

        # Hyperparam tuning
        self.ITERATIONS = 512
        self.PARAM_OPT_RESULTS_DIR = os.path.join(self.RESULTS_DIR, "ray-results")
        if not os.path.exists(self.PARAM_OPT_RESULTS_DIR):
            os.makedirs(self.PARAM_OPT_RESULTS_DIR)

        # Model training
        self.MODEL_DIR = os.path.join(self.RESULTS_DIR, "training")
        if not os.path.exists(self.MODEL_DIR):
            os.makedirs(self.MODEL_DIR)
