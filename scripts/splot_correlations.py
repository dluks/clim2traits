#!/usr/bin/env python3
import json
from typing import Optional, Union

import pandas as pd

from utils.geodata import splot_correlations


def create_correlation_table() -> pd.DataFrame:
    """Create a table for storing correlations between trait predictions and their
    corresponding sPlot grids."""
    with open("trait_mapping.json", encoding="utf-8") as f:
        trait_mapping = json.load(f)

    traits = []
    for trait in trait_mapping:
        traits.append(f"X{trait}")

    mi_columns = pd.MultiIndex.from_product(
        [["2", "0.5", "0.2", "0.01"], ["GBIF", "sPlot"]]
    )
    mi_rows = pd.MultiIndex.from_product(
        [traits, ["Grass", "Shrub-Tree", "Shrub-Tree-Grass"]]
    )
    df = pd.DataFrame(index=mi_rows, columns=mi_columns)
    df.to_parquet("results/trait_correlations.parquet")

    return df


def update_correlation_table(
    df: Optional[pd.DataFrame] = None,
    pfts: Optional[list[str]] = None,
    resolutions: Optional[list[Union[int, float]]] = None,
) -> pd.DataFrame:
    """Calculate correlations between trait predictions and their corresponding sPlot
    grids and store them in a table."""
    if df is None:
        df = pd.read_parquet("results/trait_correlations.parquet")

    if pfts is None:
        pfts = ["Grass", "Shrub_Tree", "Shrub_Tree_Grass"]

    if resolutions is None:
        resolutions = [2, 0.5, 0.2, 0.01]

    for pft in pfts:
        print(f"\n{pft}")
        for resolution in resolutions:
            gbif, splot = splot_correlations(resolution, 0.5, pft)

            for i, trait_set in enumerate([gbif, splot]):
                src = "GBIF" if i == 0 else "sPlot"
                for trait_id, correlation in trait_set.items():
                    df.loc[
                        (trait_id, pft.replace("_", "-")), (str(resolution), src)
                    ] = correlation
    return df


def main():
    """Run the script."""
    df = create_correlation_table()
    df = update_correlation_table(df)
    df.to_parquet("results/trait_correlations.parquet")


if __name__ == "__main__":
    main()
