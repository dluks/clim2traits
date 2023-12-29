from typing import Any

import geopandas as gpd
import pytest
import xarray as xr
from pandas.testing import assert_frame_equal

from ..utils.geodata import xr_to_gdf


@pytest.fixture(name="xr_da")
def fixture_xr_da() -> xr.DataArray:
    """
    Returns a sample xr.DataArray object with coordinates and dimensions.

    Returns:
        xr.DataArray: A sample xr.DataArray object.
    """
    return xr.DataArray(
        [[1, 2], [3, 4]],
        coords={"x": [0, 1], "y": [0, 1]},
        dims=["x", "y"],
        name="data",
    ).rio.write_crs("EPSG:4326")


@pytest.fixture(name="xr_ds")
def fixture_xr_ds(xr_da: xr.DataArray) -> xr.Dataset:
    """
    Create a new xarray Dataset from a given xarray DataArray.

    Parameters:
        xr_da (xr.DataArray): The input xarray DataArray.

    Returns:
        xr.Dataset: The new xarray Dataset with two variables, 'var1' and 'var2',
                    both copied from the input DataArray. The Dataset is also
                    assigned the CRS (Coordinate Reference System) 'EPSG:4326'.
    """
    return xr.Dataset({"var1": xr_da.copy(), "var2": xr_da.copy()}).rio.write_crs(
        "EPSG:4326"
    )


@pytest.fixture(name="single_var_gdf")
def fixture_single_var_gdf(crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Create a GeoDataFrame with a single variable.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with a single variable and a geometry column.
    """
    return gpd.GeoDataFrame(  # type: ignore
        {"data": [1, 2, 3, 4]},
        geometry=gpd.points_from_xy([0, 0, 1, 1], [0, 1, 0, 1]),
        crs=crs,
    )


@pytest.fixture(name="multi_var_gdf")
def fixture_multi_var_gdf(crs: Any = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Returns a GeoDataFrame with multiple variables.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame with variables 'var1' and 'var2' and corresponding geometry.
    """
    return gpd.GeoDataFrame(  # type: ignore
        {"var1": [1, 2, 3, 4], "var2": [1, 2, 3, 4]},
        geometry=gpd.points_from_xy([0, 0, 1, 1], [0, 1, 0, 1]),
        crs=crs,
    )


class TestXrToGdf:
    """
    Test class for the xr_to_gdf function.
    """

    def test_xr_to_gdf_with_data_array(
        self, xr_da: xr.DataArray, single_var_gdf: gpd.GeoDataFrame
    ):
        """
        Test the conversion of a DataArray to a GeoDataFrame using the xr_to_gdf function.
        """
        gdf = xr_to_gdf(xr_da)
        assert_frame_equal(gdf, single_var_gdf.set_crs(xr_da.rio.crs))

    def test_xr_to_gdf_with_dataset(
        self, xr_ds: xr.Dataset, multi_var_gdf: gpd.GeoDataFrame
    ):
        """
        Test the conversion of a DataArray to a GeoDataFrame using the xr_to_gdf function.
        """
        gdf = xr_to_gdf(xr_ds)
        assert_frame_equal(gdf, multi_var_gdf.set_crs(xr_ds.rio.crs))
