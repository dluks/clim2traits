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
