import json
import math
from pathlib import Path
from typing import Optional, Union

import cartopy.crs as ccrs
import dask.array as da
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacv
import xarray as xr
from matplotlib.axes import Axes
from matplotlib.pyplot import colorbar


def plot_raster_maps(fns: list, ncols: int):
    """Plots trait maps for each of the given trait map images

    Args:
        fns (list): List of filenames of the trait maps
        ncols (int): Number of columns desired for subplots
    """
    if len(fns) >= ncols:
        nrows = math.ceil(len(fns) / ncols)
    else:
        nrows = 1

    # Define figsize based on number of rows and columns
    figsize = (5 * ncols, 3 * nrows)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=figsize,
        subplot_kw={"projection": ccrs.Robinson()},
        tight_layout=True,
    )
    axes = axes.flatten()

    for ax, fn in zip(axes, fns):
        with xr.open_dataset(fn, engine="rasterio") as ds:
            darr = da.from_array(ds.band_data.squeeze(), chunks="auto")
            lon = ds.coords["x"].values
            lat = ds.coords["y"].values
            title = truncate_string(Path(fn).stem)

            ax.set_global()
            ax.coastlines(resolution="110m", linewidth=0.5)
            im = ax.imshow(
                darr,
                extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                transform=ccrs.PlateCarree(),
                cmap="viridis",
            )
            fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.5)
            ax.set_title(title)

    # Clean up trailing axes
    remainder = len(axes) - len(fns)
    if remainder > 0:
        for i in range(remainder):
            fig.delaxes(axes[-i - 1])


def plot_pred_cov(preds: list[xr.Dataset]) -> None:
    """Plot trait predictions and their corresonding Coefficient of Variation (CoV)"""
    nrows = len(preds)
    ncols = 2

    _, geo_axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw={"projection": ccrs.PlateCarree()},
        figsize=(12 * ncols, 5 * nrows),
        dpi=200,
        tight_layout=True,
    )

    with open("./trait_id_to_trait_name.json", "r") as f:
        mapping = json.load(f)

    for i, ds in enumerate(preds):
        trait = list(ds.data_vars)[0]
        trait_id = trait.split("_")[2]
        trait_num = trait_id.split("X")[-1]
        trait_name = f"GBIF TRY-GF: {mapping[trait_num]}"

        # Limit vmax of CoV geoaxes if CoV range is > 1
        cov_max = np.nanmax(ds["CoV"].values)

        vmax = 0.5 if cov_max >= 1 else None
        # print(trait_name, cov_max, vmax)

        geo_axes[i][0] = plot_dataset(ds, trait, geo_axes[i][0], trait_name)
        geo_axes[i][1] = plot_dataset(
            ds, "CoV", geo_axes[i][1], "Coefficient of Variation", vmax=vmax
        )

    plt.show()


def plot_dataset(
    da: Union[xr.Dataset, xr.DataArray],
    data: str,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    proj: ccrs.Projection = ccrs.PlateCarree,
    vmax: Optional[float] = None,
) -> Optional[Axes]:
    """
    Quick and dirty plot of a global rasterio data array

    Args:
        da xr.Dataset | xr.DataArray: xarray dataset or data array to be plotted
        data (str): Name of the data array to be plotted
        ax (Axes, optional): Matplotlib axes to plot on. Defaults to None.
        title (str, optional): Title of the plot. Defaults to None.
        proj (ccrs.Projection, optional): Desired projection. Defaults to
        ccrs.PlateCarree.
        **kwargs: Additional keyword arguments to be passed to the axis
    """
    cmap = sns.color_palette("rocket", as_cmap=True)

    if ax is None:
        _, ax = plt.subplots(
            figsize=(20, 15),
            subplot_kw={"projection": proj()},
            tight_layout=True,
        )

    if isinstance(da, xr.Dataset):
        da = da[data]

    lon = da.coords["x"].values
    lat = da.coords["y"].values
    title = title if title is not None else str(da.name)
    ax.set_global()  # type: ignore
    ax.coastlines(resolution="110m", linewidth=0.5)  # type: ignore

    # vmax = kwargs["vmax"] if kwargs is not None else None

    im = ax.contourf(
        lon,
        lat,
        np.squeeze(da),
        50,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmax=vmax,
    )

    # Set axis background color to very light grey
    ax.set_facecolor("#f0f0f0")

    colorbar(im, ax=ax, orientation="vertical", shrink=0.5)
    ax.set_title(title)

    ax.set_ylim([-60, 90])

    return ax


def plot_splits(skcv: spacv.SKCV, XYs: gpd.GeoSeries) -> None:
    """Plot spatial CV test splits

    Args:
        skcv (spacv.SKCV): The Spatial K-fold cross validator to use
        XYs (gpd.GeoSeries): XY geometries (coordinates)
    """
    _, ax = plt.subplots(subplot_kw={"projection": ccrs.Robinson()})
    ax.coastlines(resolution="110m", linewidth=0.5)  # type: ignore
    ax.set_global()  # type: ignore
    ax.set_title(f"N splits: {len(list(skcv.split(XYs)))}")

    for _, test in skcv.split(XYs):
        lon = XYs.iloc[test].x.values
        lat = XYs.iloc[test].y.values
        ax.plot(lon, lat, ".", markersize="0.5", alpha=1, transform=ccrs.PlateCarree())


def plot_distributions(
    df: Union[gpd.GeoDataFrame, pd.DataFrame], pdf: bool = False, num_cols=4
) -> None:
    num_plots = len(df.columns)
    num_rows = int(np.ceil(num_plots / num_cols))
    figsize = (5 * num_cols, 3 * num_rows)

    sns.set_style("whitegrid")
    _, axes = plt.subplots(
        num_rows, num_cols, figsize=figsize, tight_layout=True, dpi=200
    )
    axes = axes.flatten()

    for i, col in enumerate(df.columns):
        ax = axes[i]

        if pdf:
            # Plot probability density function
            sns.kdeplot(df[col], ax=ax)
            ax.set_xlabel("Value")
        else:
            # Plot histogram
            sns.histplot(df[col], ax=ax, bins=50)
        title = truncate_string(col)
        ax.set_title(title)

    # clean up empty subplots
    for i in range(num_plots, num_cols * num_rows):
        ax = axes[i]
        ax.set_axis_off()

    plt.show()


def truncate_string(string: str, max_len: int = 30) -> str:
    """Truncate a string to a maximum length, adding ellipses to the middle if necessary"""
    if len(string) > max_len:
        string = string[:10] + "..." + string[-10:]
    return string


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


def plot_gdf_map(gdf: gpd.GeoDataFrame, column: str) -> None:
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    # Set up the plot
    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # Add coastlines
    ax.coastlines(resolution="110m")

    # Set extent to global
    ax.set_global()

    # Plot the GeoDataFrame as a raster map
    gdf.plot(ax=ax, column=column, cmap="OrRd", legend=True)

    # Show the plot
    plt.show()
