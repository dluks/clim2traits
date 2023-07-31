import os
import pathlib

import numpy as np

from utils.datasets import CollectionName
from utils.training import TrainingConfig


# Configuration settings for 2-TrainModel.ipynb
class TrainModelConfig:
    def __init__(self, debug=False):
        self.TEST_MODE = False
        self.EXPLORE_SPLITS = False
        self.SAVE_AUTOCORRELATION_RANGES = False
        self.RNG_STATE = 42
        self.TRAIN_MODE = True
        self.DEBUG = debug

        # iNaturalist
        self.iNat_dir = pathlib.Path("./iNaturalist_traits/maps_iNaturalist").absolute()
        self.iNat_name = CollectionName.INAT
        self.splot_dir = pathlib.Path("./iNaturalist_traits/maps_sPlotOpen").absolute()

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

        # Model training hyperparameters
        params = {
            "n_estimators": 501,
            "max_depth": 9,
            "subsample": 0.65,
            "colsample_bytree": 0.36,
            "colsample_bylevel": 0.305,
            "learning_rate": 0.01,
        }

        # NaN strategy should be one of the following to maintain consistency in the
        # results table:
        # ["all", "all-imputed", "any", "threshold-#", "threshold-#-imputed"]
        nan_strategy = "all"

        # Define TrainingConfig
        self.training_config = TrainingConfig(
            nan_strategy=nan_strategy,
            train_test_split=0.2,
            cv_grid_size=self.AUTOCORRELATION_RANGE,
            cv_n_groups=10,
            cv_block_buffer=0.0,
            search_n_trials=200,
            optimizer="hyperopt",
            params=params,
            max_iters=1,
            n_jobs=-1,
            results_dir=self.RESULTS_DIR,
            results_csv=pathlib.Path(self.RESULTS_DIR, self.csv_fname),
            random_state=self.RNG_STATE,
            filter_y_outliers=[0.01, 0.99],
        )

        if self.DEBUG:
            # Define test mode TrainingConfig
            self.training_config.cv_n_groups = 2
            self.training_config.search_n_trials = 2
            self.training_config.results_dir = self.RESULTS_DIR / "test"
            self.training_config.results_dir.mkdir(parents=True, exist_ok=True)
            self.training_config.results_csv = (
                self.training_config.results_dir / f"test-{self.csv_fname}"
            )
