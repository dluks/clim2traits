import os

import numpy as np

from utils.datasets import CollectionName


# Configuration settings for 2-TrainModel.ipynb
class TrainModelConfig:
    def __init__(self):
        # iNaturalist
        self.iNat_dir = os.path.abspath("./iNaturalist_traits/maps_iNaturalist")
        self.iNat_name = CollectionName.INAT

        # WorldClim
        self.WC_dir = os.path.abspath("./data/worldclim/bio/")
        self.WC_bio_ids = ["1", "4", "7", "12", "13", "14", "13-14", "15"]
        self.WC_name = CollectionName.WORLDCLIM

        # MODIS
        self.MODIS_dir = os.path.abspath("./data/modis/")
        self.MODIS_name = CollectionName.MODIS

        # Soil
        self.soil_dir = os.path.abspath("./data/soil")
        self.soil_name = CollectionName.SOIL

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
