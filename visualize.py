import math
import os

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
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
