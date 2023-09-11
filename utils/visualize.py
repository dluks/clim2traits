import math
import os
from typing import Union

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacv
import xarray as xr

# import pandas as pd
# from matplotlib.colors import BoundaryNorm
# from matplotlib.ticker import MaxNLocator


def plot_traits(fns: list, ncols: int):
    """Plots trait maps for each of the given trait map images

    Args:
        fns (list): List of filenames of the trait maps
        ncols (int): Number of columns desired for subplots
    """
    # Calculate number of rows based on number of columns provided
    if len(fns) >= ncols:
        nrows = math.ceil(len(fns) / ncols)
    else:
        nrows = 1

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(20, 15),
        subplot_kw={"projection": ccrs.Robinson()},
        tight_layout=True,
    )
    axes = axes.flatten()

    for ax, fn in zip(axes, fns):
        da = xr.open_dataset(fn, engine="rasterio")
        lon = da.coords["x"].values
        lat = da.coords["y"].values
        title = os.path.basename(fn).split(".tif")[0]
        ax.set_global()
        ax.coastlines(resolution="110m", linewidth=0.5)
        im = ax.contourf(
            lon, lat, np.squeeze(da.band_data), 50, transform=ccrs.PlateCarree()
        )
        fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.5)
        ax.set_title(title)

    # Clean up trailing axes
    remainder = len(axes) - len(fns)
    if remainder > 0:
        for i in range(remainder):
            fig.delaxes(axes[-i - 1])


# Taken from https://sojwolf.github.io/iNaturalist_traits/Chapter_6_Compare_trait_maps_sPlot_iNat.html#visualize-trait-maps
# def plot_grid(df, lon, lat, variable, dataset_name, deg, log=True):
#     plt.rcParams.update({"font.size": 15})

#     # define raster shape for plotting
#     step = int((360 / deg) + 1)
#     bins_x = np.linspace(-180, 180, step)
#     bins_y = np.linspace(-90, 90, int(((step - 1) / 2) + 1))

#     df["x_bin"] = pd.cut(df[lon], bins=bins_x)
#     df["y_bin"] = pd.cut(df[lat], bins=bins_y)

#     df["x_bin"] = df["x_bin"].apply(lambda x: x.left)
#     df["y_bin"] = df["y_bin"].apply(lambda x: x.left)

#     grouped_df = df.groupby(["x_bin", "y_bin"], as_index=False)[variable].mean()
#     raster = grouped_df.pivot("y_bin", "x_bin", variable)

#     # data format
#     data_crs = ccrs.PlateCarree()

#     # for colorbar
#     levels = MaxNLocator(nbins=15).tick_values(
#         grouped_df[variable].min(), grouped_df[variable].max()
#     )
#     cmap = plt.get_cmap("YlGnBu")  # colormap
#     norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
#     im_ratio = raster.shape[0] / raster.shape[1]  # for size of colorbar

#     # create base plot of a world map
#     ax = fig.add_subplot(
#         1, 1, 1, projection=ccrs.Robinson()
#     )  # I used the PlateCarree projection from cartopy
#     ax.set_global()

#     # add grid with values
#     im = ax.pcolormesh(
#         bins_x,
#         bins_y,
#         raster,
#         cmap="YlGnBu",
#         vmin=grouped_df[t].min(),
#         vmax=grouped_df[t].max(),
#         transform=data_crs,
#     )

#     # add color bar
#     if log == True:
#         label = "log " + str(t)
#     else:
#         label = str(t)

#     fig.colorbar(im, fraction=0.046 * im_ratio, pad=0.04, label=label)

#     # add coastlines
#     ax.coastlines(resolution="110m", color="pink", linewidth=1.5)

#     # set title
#     ax.set_title(variable + " " + dataset_name, size=14)


def plot_rasterio(da: xr.DataArray, proj: ccrs.Projection = ccrs.PlateCarree):
    """
    Quick and dirty plot of a global rasterio data array

    Args:
        da (xr.DataArray): Rasterio data array to be plotted
        proj (ccrs.Projection, optional): Desired projection. Defaults to
        ccrs.PlateCarree.
    """
    fig, ax = plt.subplots(
        figsize=(20, 15),
        subplot_kw={"projection": proj()},
        tight_layout=True,
    )

    lon = da.coords["x"].values
    lat = da.coords["y"].values
    title = da.name
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.5)
    im = ax.contourf(lon, lat, np.squeeze(da), 50, transform=ccrs.PlateCarree())
    fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.5)
    ax.set_title(title)


def plot_splits(skcv: spacv.SKCV, XYs: gpd.GeoSeries) -> None:
    """Plot spatial CV test splits

    Args:
        skcv (spacv.SKCV): The Spatial K-fold cross validator to use
        XYs (gpd.GeoSeries): XY geometries (coordinates)
    """
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines(resolution="110m", linewidth=0.5)
    ax.set_global()
    ax.set_title(f"N splits: {len(list(skcv.split(XYs)))}")

    for _, test in skcv.split(XYs):
        lon = XYs.iloc[test].x.values
        lat = XYs.iloc[test].y.values
        ax.plot(lon, lat, ".", markersize="0.5", alpha=1, transform=ccrs.PlateCarree())


def plot_distributions(df: Union[gpd.GeoDataFrame, pd.DataFrame], num_cols=4) -> None:
    num_plots = len(df.columns)
    num_cols = num_cols
    num_rows = int(np.ceil(num_plots / num_cols))
    _, axes = plt.subplots(
        num_rows, num_cols, figsize=(20, 20), tight_layout=True, dpi=200
    )
    for i, col in enumerate(df.columns):
        ax = axes[i // num_cols, i % num_cols]
        ax.hist(df[col], bins=50)
        ax.set_title(col)

    # clean up empty subplots
    for i in range(num_plots, num_cols * num_rows):
        ax = axes[i // num_cols, i % num_cols]
        ax.set_axis_off()

    plt.show()


def plot_observed_vs_predicted(ax, observed, predicted, name, log: bool = False):
    """Plot observed vs. predicted values."""

    # plot the observed vs. predicted values using seaborn
    sns.set_theme()
    sns.set_style("whitegrid")

    p1 = min(min(predicted), min(observed))
    p2 = max(max(predicted), max(observed))
    if log:
        ax.loglog([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
    else:
        ax.plot([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)

    ax.scatter(predicted, observed, alpha=0.15)
    sns.kdeplot(x=predicted, y=observed, ax=ax, cmap="plasma", fill=True)

    # Fit a regression line for observed vs. predicted values, plot the regression
    # line so that it spans the entire plot, and print the correlation coefficient
    m, b = np.polyfit(predicted, observed, 1)
    reg_line = [m * p1 + b, m * p2 + b]
    if log:
        ax.loglog([p1, p2], reg_line, color="red", lw=0.5)
    else:
        ax.plot([p1, p2], reg_line, color="red", lw=0.5)
    ax.text(
        0.05,
        0.95,
        f"r = {np.corrcoef(predicted, observed)[0, 1]:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
    )

    # include legend items for the reg_line and the 1-to-1 line
    ax.legend(
        [
            ax.get_lines()[0],
            ax.get_lines()[1],
        ],
        ["1-to-1", "Regression"],
        loc="lower right",
    )

    # set informative axes and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(name)

    return ax


def plot_all_trait_obs_pred(trait_dirs, mapping=None):
    # Plot observed vs. predicted for each GBIF trait as subplots of a single figure
    # Number of subplots should equal number of GBIF traits, with 4 columns
    num_traits = len(trait_dirs)
    num_cols = 4
    num_rows = int(np.ceil(num_traits / num_cols))

    fig, axs = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(20, 5 * num_rows),
        tight_layout=True,
        dpi=200,
    )
    axs = axs.flatten()

    for i, trait_dir in enumerate(trait_dirs):
        trait = trait_dir.stem

        if mapping:
            # Update trait name to match the mapping
            trait_id = trait.split("_")[2].split("X")[1]
            trait_set = trait.split("_")[0]
            trait = f"{trait_set} {mapping[trait_id]}"

        obs_vs_pred = pd.read_parquet(trait_dir / "cv_predictions.parq")
        axs[i] = plot_observed_vs_predicted(
            axs[i], obs_vs_pred["observed"], obs_vs_pred["predicted"], trait
        )

    # Clean up empty subplots
    for i in range(num_traits, num_rows * num_cols):
        fig.delaxes(axs[i])
