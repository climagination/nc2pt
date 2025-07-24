import logging
from typing import Dict, Callable

import pandas as pd
import xarray as xr

from nc2pt.climatedata import ClimateData, ClimateModel, ClimateVariable
from nc2pt.computations import compute_normalization, compute_standardization, user_defined_transform
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
                        var: ClimateVariable,
                        climdata: ClimateData,
                        hr_ref: xr.DataArray) -> xr.DataArray:
    """
    Crop the dataset to the configured start and end times.

    Parameters
    ----------
    ds : xr.DataArray
        The input dataset.
    var : ClimateVariable
        Unused in this function; included for interface compatibility.
    model : ClimateModel
        Unused in this function; included for interface compatibility.
    climdata : ClimateData
        Global configuration object with time selection settings.
    hr_ref : xr.DataArray
        Unused in this function; included for interface compatibility.

    Returns
    -------
    xr.DataArray
        Time-cropped dataset.
    """
    start = climdata.select.time.range.start
    end = climdata.select.time.range.end
    ds = slice_time(ds, start, end)
    return ds


def apply_regrid(ds: xr.DataArray,
                 var: ClimateVariable,
                 model: ClimateModel,
                 climdata: ClimateData,
                 hr_ref: xr.DataArray) -> xr.DataArray:
    """
    Regrid the dataset to match a high-resolution reference field.

    Parameters
    ----------
    ds : xr.DataArray
        The input dataset to regrid.
    var : ClimateVariable
        Unused in this function; included for interface compatibility.
    model : ClimateModel
        The climate model configuration.
    climdata : ClimateData
        Unused in this function; included for interface compatibility.
    hr_ref : xr.DataArray
        High-resolution reference dataset.

    Returns
    -------
    xr.DataArray
        Regridded dataset.

    Raises
    ------
    ValueError
        If no hr_ref is provided in the model configuration.
    """
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
                       var: ClimateVariable,
                       model: ClimateModel,
                       climdata: ClimateData,
                       hr_ref: xr.DataArray) -> xr.DataArray:
    """
    Crop the dataset spatially to configured x/y index bounds.

    Parameters
    ----------
    ds : xr.DataArray
        The input dataset to crop.
    var : ClimateVariable
        Unused in this function; included for interface compatibility.
    model : ClimateModel
        Unused in this function; included for interface compatibility.
    climdata : ClimateData
        Global configuration containing crop index bounds.
    hr_ref : xr.DataArray
        Unused in this function; included for interface compatibility.

    Returns
    -------
    xr.DataArray
        Cropped dataset.
    """
    scale_factor = climdata.select.spatial.scale_factor
    x_range = climdata.select.spatial.x
    y_range = climdata.select.spatial.y
    ds = crop_field(ds, scale_factor, x_range, y_range)
    logging.info(f"🌎 Cropped field to x:[{x_range.first_index},{x_range.last_index}] and y:[{y_range.first_index}, {y_range.last_index}]")
    return ds


def apply_coarsen(ds: xr.DataArray,
                  var: ClimateVariable,
                  model: ClimateModel,
                  climdata: ClimateData,
                  hr_ref: xr.DataArray) -> xr.DataArray:
    """
    Coarsen the dataset spatially by a configured scale factor.

    Parameters
    ----------
    ds : xr.DataArray
        The input dataset to coarsen.
    var : ClimateVariable
        Unused in this function; included for interface compatibility.
    model : ClimateModel
        Unused in this function; included for interface compatibility.
    climdata : ClimateData
        Global configuration containing coarsening parameters.
    hr_ref : xr.DataArray
        Unused in this function; included for interface compatibility.

    Returns
    -------
    xr.DataArray
        Coarsened dataset.
    """
    scale_factor = climdata.select.spatial.scale_factor
    coarsen_lr(ds, scale_factor)
    logging.info(f"🪛  Coarsened field by a factor of {scale_factor}")
    return ds


def apply_user_defined_transforms(ds: xr.DataArray,
                                  var: ClimateVariable,
                                  model: ClimateModel,
                                  climdata: ClimateData,
                                  hr_ref: xr.DataArray) -> xr.DataArray:
    """
    Apply user-defined transformations to the input dataset.

    Parameters
    ----------
    ds : xr.DataArray
        The input dataset to transform.
    var : ClimateVariable
        Contains transformation instructions (e.g., unit conversions or log-scaling).
    model : ClimateModel
        Unused in this function; included for interface compatibility.
    climdata : ClimateData
        Unused in this function; included for interface compatibility.
    hr_ref : xr.DataArray
        Unused in this function; included for interface compatibility.

    Returns
    -------
    xr.DataArray
        Transformed dataset.
    """
    return user_defined_transform(ds, var)


def apply_data_split(ds: xr.DataArray,
                     var: ClimateVariable,
                     model: ClimateModel,
                     climdata: ClimateData,
                     hr_ref: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Split the dataset into training, validation, and test sets based on years.

    Parameters
    ----------
    ds : xr.DataArray
        The input dataset to split.
    var : ClimateVariable
        Unused in this function; included for interface compatibility.
    model : ClimateModel
        Unused in this function; included for interface compatibility.
    climdata : ClimateData
        Global configuration containing year splits.
    hr_ref : xr.DataArray
        Unused in this function; included for interface compatibility.

    Returns
    -------
    dict[str, xr.DataArray]
        Dictionary containing split datasets keyed by "train", "test", and "validation".
    """
    test_years = climdata.select.time.test_years
    validation_years = climdata.select.time.validation_years
    full_ds = train_test_split(ds, test_years, validation_years)
    logging.info(f"🪓  Split dataset into test years: {test_years}, validation years: {validation_years}, and training years (remainder)")
    return full_ds


def run_alignment_pipeline(ds: xr.DataArray,
                           var: ClimateVariable,
                           model: ClimateModel,
                           climdata: ClimateData,
                           hr_ref: xr.DataArray) -> dict[str, xr.DataArray]:
    """
    Execute the alignment pipeline for a given climate model.

    Parameters
    ----------
    ds : xr.DataArray
        Input dataset to process.
    var : ClimateVariable
        Climate variable containing custom transformation proceedures.
    model : ClimateModel
        Climate model containing the alignment pipeline steps.
    climdata : ClimateData
        Global configuration for the preprocessing run.
    hr_ref : xr.DataArray
        Optional high-resolution reference field for regridding.

    Returns
    -------
    dict[str, xr.DataArray]
        Processed dataset(s), returned as a dictionary.
        If no split is performed, a single entry with key "full" is returned.

    Raises
    ------
    ValueError
        If an unknown step is found in the model's pipeline.
    """
    for step in model.alignment_pipeline:
        if step not in alignment_steps:
            raise ValueError(f"Unknown alignment step '{step}' in model '{model.name}'")
        ds = alignment_steps[step](ds, var, model, climdata, hr_ref)

    if not isinstance(ds, dict):
        ds = {"full": ds}
    return ds


alignment_steps: dict[str, Callable] = {
    "temporal_crop": apply_temporal_crop,
    "regrid": apply_regrid,
    "spatial_crop": apply_spatial_crop,
    "coarsen": apply_coarsen,
    "user_defined_transforms": apply_user_defined_transforms,
    "split_data": apply_data_split,
}


def apply_feature_scaling(
    ds_dict: dict[str, xr.Dataset],
    var: ClimateVariable
) -> dict[str, xr.Dataset]:
    """
    Applies standardization or normalization to datasets in a dictionary.
    If train/test/validation keys are provided, uses training statistics
    for all splits. If only 'full' is present, scales in-place.

    Parameters
    ----------
    ds_dict : dict[str, xr.Dataset]
        Dictionary containing datasets to be scaled. Must contain either:
        - 'train', 'test', 'validation', or
        - 'full'
    var : ClimateVariable
        Variable config containing scaling flags and variable name.

    Returns
    -------
    dict[str, xr.Dataset]
        Same keys as input, with values scaled according to the config.
    """
    varname = var.name

    if var.apply_standardize:
        logging.info(f"📏 Standardizing {varname}...")
        if "train" in ds_dict:
            train = compute_standardization(ds_dict["train"], varname)
            test = compute_standardization(ds_dict["test"], varname, ds_dict["train"])
            val = compute_standardization(ds_dict["validation"], varname, ds_dict["train"])
            return {"train": train, "test": test, "validation": val}
        elif "full" in ds_dict:
            full = compute_standardization(ds_dict["full"], varname)
            return {"full": full}

    if var.apply_normalize:
        logging.info(f"📐 Normalizing {varname}...")
        if "train" in ds_dict:
            train = compute_normalization(ds_dict["train"], varname)
            test = compute_normalization(ds_dict["test"], varname, ds_dict["train"])
            val = compute_normalization(ds_dict["validation"], varname, ds_dict["train"])
            return {"train": train, "test": test, "validation": val}
        elif "full" in ds_dict:
            full = compute_normalization(ds_dict["full"], varname)
            return {"full": full}

    logging.info(f"🚫 Skipping standardization and normalization for {varname}")
    return ds_dict
