import os
import pathlib

import numpy as np

from utils.datasets import CollectionName
from utils.training import TrainingConfig


# Configuration settings for 2-TrainModel.ipynb
class TrainModelConfig:
    def __init__(self):
        self.TEST_MODE = False
        self.EXPLORE_SPLITS = False
        self.SAVE_AUTOCORRELATION_RANGES = False
        self.RNG_STATE = 100919
        self.TRAIN_MODE = True
        self.DEBUG = False

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

        # Spatial CV
        DEGREE = 111325  # Standard value for 1 degree in meters at the equator
        if os.path.exists("./ranges.npy"):
            self.AUTOCORRELATION_RANGE = np.median(np.load("ranges.npy")) / DEGREE
        else:
            raise ValueError("Cannot locate autocorrelation ranges")

        UPPER_BOUND = np.round(DEGREE * 120)
        STEP = np.round(UPPER_BOUND / 30)
        self.BW = STEP // 2
        self.LAGS = np.arange(0, UPPER_BOUND, STEP)

        # TRAINING
        self.RESULTS_DIR = pathlib.Path("./results").absolute()

        # make the directory if it doesn't exist
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Hyperparam tuning
        self.ITERATIONS = 512
        # self.HYPEROPT_RESULTS_DIR = pathlib.Path(self.RESULTS_DIR, "ray-results")
        # self.HYPEROPT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        # Model training
        self.MODEL_DIR = pathlib.Path(self.RESULTS_DIR, "training")
        self.csv_fname = "training_results.csv"

        # make the directory if it doesn't exist
        self.MODEL_DIR.mkdir(parents=True, exist_ok=True)

        # Define TrainingConfig
        self.training_config = TrainingConfig(
            train_test_split=0.2,
            cv_grid_size=self.AUTOCORRELATION_RANGE,
            cv_n_groups=10,
            cv_block_buffer=0.01,
            search_n_trials=200,
            n_jobs=-1,
            results_dir=self.RESULTS_DIR,
            results_csv=pathlib.Path(self.RESULTS_DIR, self.csv_fname),
            random_state=self.RNG_STATE,
        )

        if self.TEST_MODE:
            # Define test mode TrainingConfig
            self.training_config = TrainingConfig(
                train_test_split=0.2,
                cv_grid_size=self.AUTOCORRELATION_RANGE,
                cv_n_groups=2,
                search_n_trials=2,
                n_jobs=-1,
                results_dir=self.RESULTS_DIR / "test",
                results_csv=pathlib.Path(self.RESULTS_DIR, self.csv_fname),
                random_state=self.RNG_STATE,
            )
