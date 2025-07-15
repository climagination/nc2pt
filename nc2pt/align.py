import logging
from typing import Dict, Callable

import pandas as pd
import xarray as xr

from nc2pt.climatedata import ClimateData, ClimateModel
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

    # if downsample:
    #     # Optional coarsen for low resolution invariant fields.
    #     logging.info("🌎 Coarsening HR invariant to become LR invariant...")
    #     ds = coarsen_lr(ds, scale_factor)

    return ds


def interpolate(ds: xr.Dataset, grid: xr.Dataset) -> xr.Dataset:
    """Regrid and interpolate input dataset to the given grid.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to regrid and align (i.e. ERA5).
    grid : xarray.Dataset
        Grid to regrid to (i.e. wrf).

    Returns
    -------
    ds : xarray.Dataset
        Dataset regridded and aligned to the given grid.
    """
    # Check that inputs are xarray datasets
    if not isinstance(ds, xr.Dataset) and not isinstance(ds, xr.DataArray):
        raise ValueError("Input dataset is not an xarray dataset.")
    if not isinstance(grid, xr.Dataset) and not isinstance(grid, xr.DataArray):
        raise ValueError("Grid is not an xarray dataset.")

    # Check that the grid has the correct dimensions
    if "rlon" not in grid.dims or "rlat" not in grid.dims:
        raise ValueError("rlon or rlat not in grid dims, check grid")
    # Check that the dataset has the correct dimensions
    if "lon" not in ds.coords:
        raise ValueError("lon not in dataset dims, check dataset")
    if "lat" not in ds.coords:
        raise ValueError("lat not in dataset dims, check dataset")

    if "lon" not in grid.coords:
        raise ValueError("lon not in grid dims, check grid")
    if "lat" not in grid.coords:
        raise ValueError("lat not in grid dims, check grid")

    # Check that the dataset has the correct variables
    interp_points = xr.Dataset({
        "lat": grid.lat,
        "lon": grid.lon,
    })
    ds = ds.interp(
        lat=interp_points["lat"],
        lon=interp_points["lon"],
        method="linear"
    )
    return ds


def align_grid(
    ds: xr.DataArray, hr_ref: xr.DataArray, climdata: ClimateData
) -> xr.DataArray:
    """Processes low resolution dataset for machine learning.

    Regrids, aligns, crops, and coarsens the low resolution dataset for use in machine learning.

    Args:
    ds: Low resolution xarray dataset.
    climdata: Climate data configuration object.

    Returns:
    Processed low resolution xarray dataset.
    """
    # Regrid and align the dataset.
    logging.info("🚀 Regridding and interpolating...")
    ds = interpolate(ds, hr_ref)

    # Crop the field to the given size.
    logging.info("🌎 Cropping field...")
    ds = crop_field(
        ds,
        climdata.select.spatial.scale_factor,
        climdata.select.spatial.x,
        climdata.select.spatial.y,
    )
    ds = ds.drop_vars(["lat", "lon"])

    return ds


def align_with_lr(ds, hr_ref, climdata) -> xr.Dataset:
    """Processes low resolution dataset for machine learning.

    Regrids, aligns, crops, and coarsens the low resolution dataset for use in machine learning.

    Args:
    ds: Low resolution xarray dataset.
    climdata: Climate data configuration object.

    Returns:
    Processed low resolution xarray dataset.
    """
    # Regrid and align the dataset.
    ds = align_grid(ds, hr_ref, climdata)

    # Coarsen the low resolution dataset.
    logging.info("🌎 Coarsening low resolution dataset...")
    ds = coarsen_lr(ds, climdata.select.spatial.scale_factor)

    return ds


def coarsen_lr(ds: xr.DataArray, scale_factor: int) -> xr.DataArray:
    """Coarsen the low resolution dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to coarsen.
    scale_factor : int
        Scale factor of the dataset.

    Returns
    -------
    ds : xarray.Dataset
        Coarsened dataset.
    """

    ds = ds.coarsen(rlon=scale_factor, rlat=scale_factor).mean()

    return ds


def apply_temporal_crop(ds: xr.DataArray,
                        model: ClimateModel,
                        climdata: ClimateData,
                        hr_ref: xr.DataArray) -> xr.DataArray:
    start = climdata.select.time.range.start
    end = climdata.select.time.range.end
    ds = slice_time(ds, start, end)
    return ds


def apply_regrid(ds: xr.DataArray,
                 model: ClimateModel,
                 climdata: ClimateData,
                 hr_ref: xr.DataArray) -> xr.DataArray:
    logging.info("🚀 Regridding and interpolating...")

    if hr_ref is None:
        raise ValueError(
                f"Cannot perform 'regrid' step for model '{model.name}': "
                f"'hr_ref' (the high-resolution reference field) is not defined.\n"
                f"To enable regridding, add an 'hr_ref' field to the model YAML."
                )
    ds = interpolate(ds, hr_ref)
    return ds


def apply_spatial_crop(ds: xr.DataArray,
                       model: ClimateModel,
                       climdata: ClimateData,
                       hr_ref: xr.DataArray) -> xr.DataArray:
    scale_factor = climdata.select.spatial.scale_factor
    x_range = climdata.select.spatial.x
    y_range = climdata.select.spatial.y
    ds = crop_field(ds, scale_factor, x_range, y_range)
    logging.info(f"🌎 Cropped field to x:[{x_range.first_index},{x_range.last_index}] and y:[{y_range.first_index}, {y_range.last_index}]")
    return ds


def apply_coarsen(ds: xr.DataArray,
                  model: ClimateModel,
                  climdata: ClimateData,
                  hr_ref: xr.DataArray) -> xr.DataArray:
    scale_factor = climdata.select.spatial.scale_factor
    coarsen_lr(ds, scale_factor)
    logging.info(f"🪛  Coarsened field by a factor of {scale_factor}")
    return ds


def apply_data_split(ds: xr.DataArray,
                     model: ClimateModel,
                     climdata: ClimateData,
                     hr_ref: xr.DataArray) -> xr.DataArray:
    test_years = climdata.select.time.test_years
    validation_years = climdata.select.time.validation_years
    full_ds = train_test_split(ds, test_years, validation_years)
    logging.info(f"🪓  Split dataset into test years: {test_years}, validation years: {validation_years}, and training years (remainder)")

    return full_ds


def run_alignment_pipeline(ds: xr.DataArray,
                           model: ClimateModel,
                           climdata: ClimateData,
                           hr_ref: xr.DataArray) -> dict[str, xr.DataArray]:
    for step in model.alignment_pipeline:
        if step not in alignment_steps:
            raise ValueError(f"Unknown alignment step '{step}' in model '{model.name}'")
        ds = alignment_steps[step](ds, model, climdata, hr_ref)
    
    # ensure that the output of this function is a dict, even if the dataset
    # has not been split
    if not isinstance(ds, dict):
        ds = {
            "full": ds
        }


    return ds


alignment_steps: dict[str, Callable] = {
    "temporal_crop": apply_temporal_crop,
    "regrid": apply_regrid,
    "spatial_crop": apply_spatial_crop,
    "coarsen": apply_coarsen,
    "split_data": apply_data_split,
}
