from utils.data_retrieval import get_fns


# Configuration settings for 1-Preprocessing.ipynb
class PreprocessingConfig:
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

        self.plot_traits = False

        # Enable to run the GEE exports
        self.gee_export = False
        # Enable to run the resampling operations that write to disk
        self.resamp_to_disk = False

    # def iNat_fns(self, res: str = "0.5", transform: str = "ln") -> list[str]:
    #     fns = utils.iNat_fns(self.iNat_dir, res, transform)
    #     return fns

    # def WC_fns(self, bios: list[int] = None, res: str = "0.5_deg") -> list[str]:
    #     """Get list of all WorldClim bio observation filenames. Takes a list of BIO
    #     variables if desired.

    #     Args:
    #         vars (list[int], optional): List of BIO variables. Returns all if not provided.
    #         Defaults to None.
    #         res (str, optional): Directory name of the desired resolution. Defaults to
    #         "0.5_deg".

    #     Returns:
    #         list[str]: List of filenames
    #     """
    #     file_dir = os.path.join(self.WC_dir, res)

    #     if not bios:
    #         fns = glob.glob(os.path.join(file_dir, "*.tif"))
    #     else:
    #         fns = list(
    #             itertools.chain.from_iterable(
    #                 [
    #                     glob.glob(f"{file_dir}/wc2.1_{res}_bio_{b}.tif", recursive=True)
    #                     for b in bios
    #                 ]
    #             )
    #         )
    #     return sorted(fns)

    # def MODIS_fns(self, res: str = "0.5_deg") -> list[str]:
    #     """Get list of monthly MODIS observation filenames by resolution.

    #     Args:
    #         res (str, optional): Directory name for desired resolution. Defaults to
    #         "0.5_deg".

    #     Returns:
    #         list[str]: List of filenames
    #     """
    #     fns = glob.glob(os.path.join(self.MODIS_dir, f"{res}", "*.tif"))
    #     return sorted(fns)

    # def soil_fns(self, res: str = "0.5_deg") -> list[str]:
    #     """Get list of soil observation filenames by resolution.

    #     Args:
    #         res (str, optional): Directory name for desired resolution. Defaults to
    #         "0.5_deg".

    #     Returns:
    #         list[str]: List of filenames
    #     """
    #     fns = glob.glob(os.path.join(self.soil_dir, "*", res, "*.tif"))
    #     return sorted(fns)
