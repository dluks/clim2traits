import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import cartopy.crs as ccrs
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import spacv
import xarray as xr
from adjustText import adjust_text
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.pyplot import colorbar
from sklearn.neighbors import NearestNeighbors

os.environ["USE_PYGEOS"] = "0"

import geopandas as gpd


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
            extent = [
                ds.coords["x"].values.min(),
                ds.coords["x"].values.max(),
                ds.coords["y"].values.min(),
                ds.coords["y"].values.max(),
            ]
            title = truncate_string(Path(fn).stem)

            ax.set_global()
            ax.coastlines(resolution="110m", linewidth=0.5)
            im = ax.imshow(
                darr,
                extent=extent,
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

    with open("./trait_id_to_trait_name.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)

    for i, ds in enumerate(preds):
        trait = list(ds.data_vars)[0]
        trait_id = trait.split("_")[2]
        trait_num = trait_id.split("X")[-1]
        trait_name = f"GBIF TRY-GF: {mapping[trait_num]}"

        # Limit vmax of CoV geoaxes if CoV range is > 1
        cov_max = np.nanmax(ds["CoV"].values)

        vmax = 0.5 if cov_max >= 1 else None

        geo_axes[i][0] = plot_dataset(
            ds,
            trait,
            AxProps(
                ax=geo_axes[i][0],
                title=trait_name,
                cmap=sns.cubehelix_palette(
                    start=2, rot=0, dark=0, light=0.95, as_cmap=True
                ),  # type: ignore
            ),
        )
        geo_axes[i][1] = plot_dataset(
            ds,
            "CoV",
            AxProps(
                ax=geo_axes[i][1],
                title=f"{trait_name} CoV",
                vmax=vmax,
                cmap=sns.color_palette("Blues", as_cmap=True),  # type: ignore
            ),
        )

    plt.show()


@dataclass
class AxProps:
    """Dataclass for storing information about a matplotlib axis"""

    ax: Optional[Axes] = None
    title: Optional[str] = None
    proj: ccrs.Projection = ccrs.PlateCarree
    cmap: ListedColormap = sns.color_palette("rocket", as_cmap=True)  # type: ignore
    vmax: Optional[float] = None


def plot_dataset(
    data: Union[xr.Dataset, xr.DataArray], data_name: str, ax_props: AxProps
) -> Optional[Axes]:
    """
    Quick and dirty plot of a global rasterio data array

    Args:
        da xr.Dataset | xr.DataArray: xarray dataset or data array to be plotted
        data (str): Name of the data array to be plotted
        ax_props (AxProps): Information about the axis to be plotted on
        **kwargs: Additional keyword arguments to be passed to the axis
    """

    if ax_props.ax is None:
        _, ax = plt.subplots(
            figsize=(20, 15),
            subplot_kw={"projection": ax_props.proj()},
            tight_layout=True,
        )

    if isinstance(data, xr.Dataset):
        data = data[data_name]

    lon = data.coords["x"].values
    lat = data.coords["y"].values
    title = title if title is not None else str(data.name)
    ax.set_global()  # type: ignore
    ax.coastlines(resolution="110m", linewidth=0.5)  # type: ignore

    im = ax.contourf(
        lon,
        lat,
        np.squeeze(data),
        50,
        transform=ccrs.PlateCarree(),
        cmap=ax_props.cmap,
        vmax=ax_props.vmax,
    )

    # Set axis background color to very light grey
    ax.set_facecolor("#f0f0f0")

    colorbar(im, ax=ax, orientation="vertical", shrink=0.5)
    ax.set_title(title, fontsize=25)

    ax.set_ylim(-60.0, 90.0)

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
    """Plot the distributions of the given dataframe's columns"""
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
            sns.histplot(df[col], ax=ax, bins=50)  # type: ignore
        title = truncate_string(col)
        ax.set_title(title)

    # clean up empty subplots
    for i in range(num_plots, num_cols * num_rows):
        ax = axes[i]
        ax.set_axis_off()

    plt.show()


def plot_spatial_distribution(
    trait_dataframe: gpd.GeoDataFrame, prediction_dataframe: gpd.GeoDataFrame, n_cols=4
):
    """Plot the spatial distribution of a GeoDataFrame as a density curve"""

    traits = trait_dataframe.columns.difference(["geometry"])

    n_plots = len(traits)
    n_rows = int(np.ceil(n_plots / n_cols))
    figsize = (5 * n_cols, 3 * n_rows)

    sns.set_style("darkgrid")
    _, axes = plt.subplots(n_rows, n_cols, figsize=figsize, tight_layout=True, dpi=200)
    axes = axes.flatten()

    for trait, ax in zip(traits, axes):
        trait_df = trait_dataframe[["geometry", trait]].dropna(subset=[trait])

        # Extract the x and y coordinates from the geometry and convert to radians
        coords_trait = np.radians(
            trait_df.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
        )
        coords_predictions = np.radians(
            prediction_dataframe.geometry.apply(lambda geom: (geom.x, geom.y)).tolist()
        )

        # Compute the nearest neighbor distances
        nbrs_trait = NearestNeighbors(n_neighbors=2, metric="haversine").fit(
            coords_trait
        )

        dist_trait, _ = nbrs_trait.kneighbors(coords_trait)
        dist_trait_predictions, _ = nbrs_trait.kneighbors(coords_predictions)

        nn_trait = dist_trait[:, 1]
        nn_trait_predictions = dist_trait_predictions[:, 1]

        # Convert the distances from radians to kilometers (assuming Earth's radius is
        # 6371 km)
        nn_trait *= 6371
        nn_trait_predictions *= 6371

        sns.kdeplot(
            [
                nn_trait,
                nn_trait_predictions,
            ],
            common_norm=True,
            common_grid=True,
            fill=True,
            log_scale=(True, False),
            legend=False,
            # bw_adjust=2,
            # palette=["red", "blue"],
            ax=ax,
        )
        ax.set_title(trait)
        # plt.xlabel("Nearest neighbor distance (km)")
        # plt.ylabel("Density")
        # plt.title("Spatial distribution of test trait values")

    # clean up empty subplots
    for i in range(n_plots, n_cols * n_rows):
        ax = axes[i]
        ax.set_axis_off()


def truncate_string(string: str, max_len: int = 30) -> str:
    """Truncate a string to a maximum length, adding ellipses to the middle if necessary"""
    if len(string) > max_len:
        string = string[:10] + "..." + string[-10:]
    return string


def plot_observed_vs_predicted(
    ax: plt.Axes,
    observed: pd.Series,
    predicted: pd.Series,
    name: str,
    log: bool = False,
    density: bool = False,
    show_r: bool = True,
):
    """Plot observed vs. predicted values."""

    # plot the observed vs. predicted values using seaborn
    sns.set_theme()

    p1 = min(predicted.min(), observed.min())
    p2 = max(predicted.max(), observed.max())

    cmap = sns.cubehelix_palette(start=0.5, rot=-0.75, reverse=True, as_cmap=True)  # type: ignore
    if density:
        sns.kdeplot(x=predicted, y=observed, ax=ax, cmap=cmap, fill=True, thresh=0.0075)
    else:
        sns.scatterplot(x=predicted, y=observed, ax=ax, s=1)

    # Fit a regression line for observed vs. predicted values, plot the regression
    # line so that it spans the entire plot, and print the correlation coefficient
    m, b = np.polyfit(predicted, observed, 1)
    reg_line = [m * p1 + b, m * p2 + b]

    if log:
        ax.loglog([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
        ax.loglog([p1, p2], reg_line, color="red", lw=0.5)
    else:
        ax.plot([p1, p2], [p1, p2], color="black", ls="-.", lw=0.5, alpha=0.5)
        ax.plot([p1, p2], reg_line, color="red", lw=0.5)

    # make sure lines are positioned on top of kdeplot
    ax.set_zorder(1)

    buffer_color = "#e9e9f1"

    if show_r:
        ax.text(
            0.05,
            0.95,
            f"$r$ = {np.corrcoef(predicted, observed)[0, 1]:.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": buffer_color, "edgecolor": buffer_color, "pad": 0.5},
        )

    ax.text(
        0.05,
        0.90,
        f"n = {len(predicted):,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        bbox={"facecolor": buffer_color, "edgecolor": buffer_color, "pad": 0.5},
    )

    # include legend items for the reg_line and the 1-to-1 line
    ax.legend(
        [
            ax.get_lines()[0],
            ax.get_lines()[1],
        ],
        ["1-to-1", "Regression"],
        loc="lower right",
        frameon=False,
    )

    # set informative axes and title
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Observed")
    ax.set_title(name)

    return ax


def plot_all_trait_obs_pred(trait_dirs, mapping=None):
    """Plot observed vs. predicted values for all traits in the given list of trait directories."""
    num_traits = len(trait_dirs)
    num_cols = 5
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

        log = trait.endswith("_ln")

        if mapping:
            # Update trait name to match the mapping
            trait_id = trait.split("_")[2].split("X")[1]
            trait_set = trait.split("_")[0]
            trait = f"{trait_set} {mapping[trait_id]['short']}"
            if log:
                trait += " (log)"

        obs_vs_pred = pd.read_parquet(trait_dir / "cv_predictions.parq")

        obs = obs_vs_pred["observed"]
        pred = obs_vs_pred["predicted"]

        axs[i] = plot_observed_vs_predicted(axs[i], obs, pred, trait, show_r=False)
        r_mean, r_std = cv_r(obs_vs_pred)
        axs[i].text(
            0.05,
            0.95,
            f"$r$ = {r_mean:.2f}±{r_std:.2f}",
            transform=axs[i].transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "#e9e9f1", "edgecolor": "#e9e9f1", "pad": 0.5},
        )

    # Ensure that only the left-most column has y-axis labels
    for i in range(num_traits):
        if i % num_cols != 0:
            axs[i].set_ylabel("")

    # Clean up empty subplots
    for i in range(num_traits, num_rows * num_cols):
        fig.delaxes(axs[i])


def cv_r(df: pd.DataFrame) -> Tuple[float, float]:
    """Calculate the mean and standard deviation of the correlation coefficient for each
    fold in the given dataframe"""
    grouped = df.groupby("fold").apply(lambda x: x["observed"].corr(x["predicted"]))
    return (grouped.mean(), grouped.std())


def plot_splot_correlations(df: pd.DataFrame, pft: str):
    """Plot sPlot correlations for GBIF and sPlot extrapolations for the given PFT"""
    idx = pd.IndexSlice
    df = df.loc[idx[:, pft], :]

    # Hide the PFT index
    df.index = df.index.droplevel(1)

    # Set the plot style
    ### Plotting configuration
    sns.set_theme(
        context="paper",
        style="ticks",
        palette="deep",
        font="FreeSans",
        font_scale=1,
        color_codes=True,
        rc=None,
    )

    # Figure directory
    fig_dir = Path("../reports/figures")
    sns.set_theme(style="whitegrid")
    custom_params = {"axes.spines.right": False, "axes.spines.top": False}
    sns.set_theme(style="ticks", rc=custom_params)

    fig, axs = plt.subplots(1, 2, figsize=(20, 25), dpi=150)
    df = df.sort_index(ascending=False)
    text_x = 0.98

    # Define colors
    # colors = plt.cm.Paired(np.linspace(0, 1, len(stg.index.get_level_values(0).unique())))
    # use sns instead (e.g. sns.hls_palette(h=.5))
    colors = sns.color_palette(n_colors=len(df.index.get_level_values(0).unique()))

    with open("trait_mapping.json", "r", encoding="utf-8") as f:
        trait_mapping = json.load(f)

    x_positions = [text_x] * len(df.index.get_level_values(0).unique())

    y_positions_gbif = []
    y_positions_splot = []

    labels = []
    label_colors = []

    # Loop over each trait
    for color, trait in zip(colors, df.index.get_level_values(0).unique()):
        # Select data for the current trait
        trait_data = df.loc[trait]

        trait_short = trait_mapping[trait.split("X")[-1]]["short"]

        # Plot GBIF data with dotted line and circular markers
        gbif_data = trait_data.xs("GBIF", axis=0, level=1)
        y_positions_gbif.append(gbif_data.iloc[0])
        labels.append(trait_short)
        label_colors.append(color)

        axs[0].plot(
            gbif_data,
            linestyle="-",
            color=color,
            label=f"{trait} GBIF",
            marker="o",
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        axs[0].set_title("GBIF")

        # Plot sPlot data with solid line and circular markers
        splot_data = trait_data.xs("sPlot", axis=0, level=1)
        y_positions_splot.append(splot_data.iloc[0])

        axs[1].plot(
            splot_data,
            linestyle="-",
            color=color,
            label=f"{trait} sPlot",
            marker="o",
            markeredgecolor="white",
            markeredgewidth=1.0,
        )
        axs[1].set_title("sPlot")

    texts_gbif = []
    for x_position, y_position, label, color in zip(
        x_positions, y_positions_gbif, labels, label_colors
    ):
        text = axs[0].text(
            x_position,
            y_position,
            label,
            ha="right",
            va="center",
            color=color,
        )
        texts_gbif.append(text)

    # add space between plots
    plt.subplots_adjust(wspace=0.5)

    # make sure the plots share the same y-axis
    axs[0].set_ylim(axs[1].get_ylim())

    texts_splot = []
    for x_position, y_position, label, color in zip(
        x_positions, y_positions_splot, labels, label_colors
    ):
        text = axs[1].text(
            x_position,
            y_position,
            label,
            ha="right",
            va="center",
            color=color,
        )
        texts_splot.append(text)

    adjust_text_kwargs = {
        "force_text": (0, 0.5),
        "only_move": {"text": "y", "static": "y", "explode": "y", "pull": "y"},
    }

    adjust_text(
        texts_gbif,
        ax=axs[0],
        **adjust_text_kwargs,
    )

    adjust_text(
        texts_splot,
        ax=axs[1],
        **adjust_text_kwargs,
    )

    # Readjust the x-position of the text since adjust_text doesn't seem to respect the
    # only_move parameter and still moves the text in the x-direction
    def _reset_text_x(texts, x_position, ax):
        for text in texts:
            text.set_ha("left")
            text.set_x(x_position)
            text.set_transform(ax.get_yaxis_transform())

    _reset_text_x(texts_gbif, text_x, axs[0])
    _reset_text_x(texts_splot, text_x, axs[1])

    # Set labels and reverse x-axis
    for ax in axs:
        ax.set_xlabel("Resolution ($\degree$)")
        ax.set_ylabel("$r$")
        ax.invert_xaxis()

    # Remove the gridlines
    axs[0].grid(False)
    axs[1].grid(False)

    # Add a title to the figure overall
    fig.suptitle(
        f"sPlot correlations for GBIF and sPlot extrapolations for {pft} PFT",
        fontsize=16,
    )

    # Show the plots
    plt.show()


def plot_hex_density(
    dataframe: Union[
        pd.DataFrame, gpd.GeoDataFrame, list[Union[pd.DataFrame, gpd.GeoDataFrame]]
    ],
    resolution: Union[int, float] = 0.5,
    ncols: int = 2,
    global_extent: bool = True,
    log: bool = False,
    names: Optional[list[str]] = None,
    label_subplots: bool = False,
) -> list[GeoAxes]:
    """Plot hex density of the given dataframe(s). Latitude and longitude columns must
    be named "lat" and "lon"."""
    if isinstance(dataframe, (pd.DataFrame, gpd.GeoDataFrame)):
        dataframe = [dataframe]

    for df in dataframe:
        if not isinstance(df, (pd.DataFrame, gpd.GeoDataFrame)):
            raise TypeError(
                "Dataframe must be of type pd.DataFrame or gpd.GeoDataFrame, not "
                f"{type(df)}"
            )
        if "lat" not in df.columns or "lon" not in df.columns:  # type: ignore
            raise ValueError(
                "Dataframe must have columns named 'lat' and 'lon' for latitude and "
                "longitude, respectively"
            )

    if len(dataframe) > 1:
        nrows, figsize = _nrows_figsize(len(dataframe), ncols)
    else:
        ncols = 1
        nrows = 1
        figsize = (15, 9)

    _, axs = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        subplot_kw={"projection": ccrs.Robinson()},
        figsize=figsize,
    )

    if len(dataframe) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, (ax, df) in enumerate(zip(axs, dataframe)):
        if global_extent:
            ax.set_global()

        if isinstance(df, gpd.GeoDataFrame):
            df = df.to_crs("EPSG:4326")  # type: ignore
            df["lon"] = df.geometry.x
            df["lat"] = df.geometry.y

        if names is not None:
            name = names[i]

        if label_subplots:
            # label the subplot with the corresponding letter a-z in the top left corner in bold
            ax.text(
                0.05,
                0.95,
                chr(97 + i),
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontweight="bold",
            )
        ax = _hexbin_ax(
            ax, df["lon"], df["lat"], gridsize=int(360 / resolution), log=log, name=name
        )

    return axs


def _hexbin_ax(
    ax: GeoAxes,
    lon: Union[pd.Series, np.ndarray],
    lat: Union[pd.Series, np.ndarray],
    gridsize: int,
    log: bool = False,
    name: Optional[str] = None,
) -> GeoAxes:
    """Plot hexbin on the given axis"""

    hb = ax.hexbin(
        lon,
        lat,
        gridsize=gridsize,
        # cmap=sns.color_palette("mako", as_cmap=True),
        cmap=sns.cubehelix_palette(
            start=0.5, rot=-0.75, dark=0.1, light=0.9, reverse=False, as_cmap=True
        ),
        mincnt=1,
        bins="log" if log else None,
        transform=ccrs.PlateCarree(),
        vmax=10000,
    )

    if name is None:
        cb_label = "Number of observations"
        title = "Global density"
    else:
        cb_label = f"Number of {name} observations"
        title = f"{name} global density"

    plt.colorbar(hb, ax=ax, label=cb_label, shrink=0.8)
    ax.coastlines(resolution="110m", linewidth=0.5)
    ax.set_title(title)

    return ax


def _nrows_figsize(data_length: int, ncols: int) -> tuple[int, tuple[int, int]]:
    """Calculate number of rows and figsize for given data length and number of columns"""
    if data_length >= ncols:
        nrows = math.ceil(data_length / ncols)
    else:
        nrows = 1

    # Define figsize based on number of rows and columns
    if ncols == 1:
        figsize = (15, 9)
    else:
        figsize = (5 * ncols, 3 * nrows)

    return nrows, figsize
