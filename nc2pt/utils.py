import xarray as xr
import pandas as pd
import logging
from typing import Dict
import omegaconf


def slice_time(ds: xr.Dataset, start: str, end: str) -> xr.Dataset:
    """Slice the time dimension of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to slice.
    start : str
        Start time.
    end : str
        End time.

    Returns
    -------
    ds : xarray.Dataset
        Sliced dataset.
    """
    # Check that end is not before start
    end = pd.to_datetime(end)
    start = pd.to_datetime(start)

    if end < start:
        raise ValueError("End date is before start date.")

    # Check if start is before the dataset
    if start < ds.time.min():
        raise ValueError("Start date is before the dataset begins.")

    # Check if end is after the dataset
    if end > ds.time.max():
        raise ValueError("End date is after the dataset ends.")

    ds = ds.sel(time=slice(start, end), drop=True)
    logging.info(f"🕜 Cropped dataset temporally from {start} to {end}")

    return ds


def train_test_split(
    ds: xr.Dataset, test_years: list, validation_years: list
) -> Dict[str, xr.Dataset]:
    """Split the dataset into a training and test set.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to split.
    years : list
        List of years to use for the test set.

    Returns
    -------
    train : xarray.Dataset
        Training dataset.
    test : xarray.Dataset
        Test dataset.
    """

    # check that list of years is not empty
    if not test_years or not validation_years:
        raise ValueError("List of years is empty.")

    # check if years is a list
    if not isinstance(test_years, list) and not isinstance(
        test_years, omegaconf.listconfig.ListConfig
    ):
        raise ValueError("Test years is not a list.")

    if not isinstance(validation_years, list) and not isinstance(
        validation_years, omegaconf.listconfig.ListConfig
    ):
        raise ValueError("Validation years is not a list.")

    test_validation_years = test_years + validation_years
    train = ds.isel(time=~ds.time.dt.year.isin(test_validation_years), drop=True)
    test = ds.isel(time=ds.time.dt.year.isin(test_years), drop=True)
    validation = ds.isel(time=ds.time.dt.year.isin(validation_years), drop=True)

    return {"train": train, "test": test, "validation": validation}


def crop_field(ds: xr.DataArray, scale_factor: int, x: dict, y: dict, downsample=False) -> xr.DataArray:
    """Crop the field to the given size.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to crop.
    scale_factor : int
        Scale factor of the dataset.
    x : OmegaConf
        Containing longitudinal spatial extent information from config.
    y : OmegaConf
        Containing latitudinal spatial extent information from config.

    Returns
    -------
    ds : xarray.Dataset
        Cropped dataset.
    """
    assert "rlon" in ds.dims, "rlon not in dims, check dataset"
    assert "rlat" in ds.dims, "rlat not in dims, check dataset"

    ds = ds.isel(
        rlon=slice(x.first_index, x.last_index),
        rlat=slice(y.first_index, y.last_index),
        drop=True,
    )

    assert (
        x.last_index - x.first_index
    ) % scale_factor == 0, "x dimension not divisible by scale factor, check config"
    assert (
        y.last_index - y.first_index
    ) % scale_factor == 0, "y dimension not divisible by scale factor, check config"

    assert (
        ds.rlon.size == ds.rlat.size
    ), "rlon and rlat not the same size, check dataset"

    return ds


def coarsen_lr(ds: xr.DataArray, scale_factor: int, method: str = 'mean') -> xr.DataArray:
    """Coarsen the low resolution dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to coarsen.
    scale_factor : int
        Scale factor of the dataset.
    method : str, optional
        Coarsening method: e.g., 'mean', 'max', 'min', 'median'.
        Default is 'mean'.

    Returns
    -------
    ds : xarray.Dataset
        Coarsened dataset.
    """
    coarsen_obj = ds.coarsen(rlon=scale_factor, rlat=scale_factor)

    # Check if the given method is a valid method of xarray's coarsen object
    if not hasattr(coarsen_obj, method):
        raise AttributeError(
            f"Unknown coarsening method: '{method}'. "
            f"Refer to xarray documentation for valid aggregation methods."
        )

    ds = getattr(coarsen_obj, method)()

    return ds
