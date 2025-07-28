import logging
from typing import Callable
import xarray as xr

from nc2pt.climatedata import ClimateData, ClimateModel, ClimateVariable
from nc2pt.computations import user_defined_transform, interpolate
from nc2pt.utils import slice_time, train_test_split, crop_field, coarsen_lr


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
