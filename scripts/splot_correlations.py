#!/usr/bin/env python3
import argparse
import json
import logging
from typing import Union

import pandas as pd

from utils.spatial_stats import splot_correlations


def create_correlation_table() -> pd.DataFrame:
    """
    Create a table for storing correlations between trait predictions and their
    corresponding sPlot grids.

    Returns:
        pd.DataFrame: The correlation table with multi-index columns and rows.
    """
    with open("trait_mapping.json", encoding="utf-8") as f:
        trait_mapping = json.load(f)

    traits = []
    for trait in trait_mapping:
        traits.append(f"X{trait}")

    mi_columns = pd.MultiIndex.from_product(
        [["2", "0.5", "0.2", "0.01"], ["GBIF", "sPlot"]],
        names=["Resolution", "Training data"],
    )
    mi_rows = pd.MultiIndex.from_product(
        [traits, ["Grass", "Shrub-Tree", "Shrub-Tree-Grass"]], names=["Trait", "PFT"]
    )
    df = pd.DataFrame(index=mi_rows, columns=mi_columns)

    return df


def update_correlation_table(
    df: pd.DataFrame,
    pfts: list[str],
    resolutions: list[Union[int, float]],
) -> pd.DataFrame:
    """
    Calculate correlations between trait predictions and their corresponding sPlot
    grids and store them in a table.

    Args:
        df (pd.DataFrame): The DataFrame to store the correlation values.
        pfts (list[str]): List of plant functional types.
        resolutions (list[Union[int, float]]): List of resolutions.

    Returns:
        pd.DataFrame: The updated DataFrame with correlation values.
    """

    for pft in pfts:
        print(f"\n{pft}")
        for resolution in resolutions:
            logging.info("Calculating correlations for %s...", resolution)
            gbif, splot = splot_correlations(resolution, 0.5, pft)

            for i, trait_set in enumerate([gbif, splot]):
                src = "GBIF" if i == 0 else "sPlot"
                for trait_id, correlation in trait_set.items():
                    df.loc[
                        (trait_id, pft.replace("_", "-")), (str(resolution), src)
                    ] = correlation
    return df


def main():
    """Main script.

    This function is the entry point of the script. It parses command line arguments,
    creates a correlation table, updates the table based on the provided arguments,
    and saves the table to a parquet file.

    Args:
        None

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_fn",
        default="results/trait_correlations.parquet",
        help="Path to output file",
    )
    parser.add_argument(
        "-p",
        "--pfts",
        nargs="+",
        default=["Grass", "Shrub_Tree", "Shrub_Tree_Grass"],
        help="PFTs to include in the table.",
    )
    parser.add_argument(
        "-r",
        "--resolutions",
        nargs="+",
        default=[2, 0.5, 0.2, 0.01],
        help="Resolutions to include in the table.",
    )
    args = parser.parse_args()

    df = create_correlation_table()
    df = update_correlation_table(df, args.pfts, args.resolutions)
    df.to_parquet(args.output_fn)


if __name__ == "__main__":
    main()
