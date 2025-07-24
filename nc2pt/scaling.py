from pathlib import Path
import xarray as xr
from typing import Optional
import logging

from nc2pt.climatedata import ClimateVariable, ClimateModel, FeatureScalingMetadata
from nc2pt.computations import compute_normalization, compute_standardization


def get_feature_scaling_stats(var: ClimateVariable, model: ClimateModel) -> dict[str, float]:
    """
    Loads and validates feature scaling metadata for a given variable.

    Parameters
    ----------
    var : ClimateVariable
        Variable configuration, including metadata path and scaling flags.
    model : ClimateModel
        Model configuration. If `emulation_data` is False, no metadata is loaded.

    Returns
    -------
    dict[str, float]
        Dictionary containing scaling statistics, e.g., {'min': ..., 'max': ...}
        or {'mean': ..., 'std': ...}. Returns an empty dict if no metadata is required.
    """
    if not model.emulation_data:
        return {}
    metadata = FeatureScalingMetadata.from_json(Path(var.metadata_path))
    metadata.validate_against_var(var)
    return metadata.get_stats()


def scale_dataset_split(
    method: str,
    ds_dict: dict[str, xr.Dataset],
    varname: str,
    stats: Optional[dict[str, float]] = None
) -> dict[str, xr.Dataset]:
    """
    Applies a selected scaling method to each split in the dataset dictionary.

    Parameters
    ----------
    method : str
        Scaling method to apply. Must be either "standardize" or "normalize".
    ds_dict : dict[str, xr.Dataset]
        Dictionary containing one of:
        - 'train', 'test', 'validation' keys
        - or a single 'full' key
    varname : str
        Name of the variable to scale within each dataset.
    stats : dict[str, float], optional
        Precomputed statistics (e.g., min/max or mean/std) for 'full' datasets.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with the same keys as `ds_dict`, but with scaled data.

    Raises
    ------
    ValueError
        If `ds_dict` doesn't contain expected keys.
    """
    compute_fn = {
        "standardize": compute_standardization,
        "normalize": compute_normalization,
    }[method]

    if "train" in ds_dict:
        train = compute_fn(ds=ds_dict["train"], varname=varname)
        test = compute_fn(ds=ds_dict["test"], varname=varname, precomputed=ds_dict["train"])
        val = compute_fn(ds_dict["validation"], varname, precomputed=ds_dict["train"])
        return {"train": train, "test": test, "validation": val}
    elif "full" in ds_dict:
        return {
            "full": compute_fn(ds_dict["full"], varname, feature_scaling_stats=stats)
        }
    else:
        raise ValueError("Expected 'train/test/validation' or 'full' in dataset dict.")


def apply_feature_scaling(
    ds_dict: dict[str, xr.Dataset],
    var: ClimateVariable,
    model: ClimateModel
) -> dict[str, xr.Dataset]:
    """
    Applies standardization or normalization to datasets in a dictionary.
    For train/test/validation splits, uses training statistics. For a 'full'
    dataset, applies scaling in-place using metadata.

    Parameters
    ----------
    ds_dict : dict[str, xr.Dataset]
        Dictionary containing datasets to be scaled. Must contain either:
        - 'train', 'test', 'validation', or
        - 'full'
    var : ClimateVariable
        Variable config containing scaling flags and variable name.
    model : ClimateModel
        Model config used to determine if metadata is required.

    Returns
    -------
    dict[str, xr.Dataset]
        Dictionary with the same keys as input, with values scaled according
        to the variable's scaling configuration.

    Raises
    ------
    ValueError
        If variable configuration is inconsistent or required keys are missing.
    """
    varname = var.name
    stats = get_feature_scaling_stats(var, model)

    if var.apply_standardize:
        logging.info(f"📏 Standardizing {varname}...")
        return scale_dataset_split("standardize", ds_dict, varname, stats)

    if var.apply_normalize:
        logging.info(f"📐 Normalizing {varname}...")
        return scale_dataset_split("normalize", ds_dict, varname, stats)

    logging.info(f"🚫 Skipping standardization and normalization for {varname}")
    return ds_dict
