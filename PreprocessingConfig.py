import glob
import itertools
import os


# Configuration settings for 1-Preprocessing.ipynb
class Config:
    def __init__(self):
        # iNaturalist
        self.iNat_dir = "./iNaturalist_traits/maps_iNaturalist"

        # WorldClim
        self.WC_dir = "./data/worldclim/bio/"

    def iNat_fns(self, res: str = "0.5", transform: str = "ln") -> list:
        """Get list of iNaturalist trait map filenames by resolution and transformation

        Args:
            res (str, optional): Resolution. Defaults to "0.5".
            transform (str, optional): Transformation ("ln" or "exp_ln"). Defaults to
            "ln".

        Returns:
            list: List of filenames
        """
        fns = glob.glob(os.path.join(self.iNat_dir, f"{res}_deg/{transform}", "*.tif"))
        return fns

    def WC_fns(self, bios: list = None) -> list:
        """Get list of all WorldClim bio observation filenames. Takes a list of BIO
        variables if desired.

        Args:
            vars (list, optional): List of BIO variables. Returns all if not provided.
            Defaults to None.

        Returns:
            list: List of filenames
        """

        if not bios:
            fns = glob.glob(os.path.join(self.WC_dir, "*.tif"))
        else:
            fns = list(
                itertools.chain.from_iterable(
                    [
                        glob.glob(
                            f"{self.WC_dir}/wc2.1_10m_bio_{b}.tif", recursive=True
                        )
                        for b in bios
                    ]
                )
            )
        return fns
