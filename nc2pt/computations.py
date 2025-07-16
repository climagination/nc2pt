import logging
import xarray as xr
import numpy as np  # noqa: F401
#from nc2pt.align import train_test_split
from nc2pt.climatedata import ClimateVariable


def user_defined_transform(ds: xr.Dataset, var: ClimateVariable) -> xr.Dataset:
    """Apply user defined transform to the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to apply transform to.
    var : ClimateVariable
        Climate variable to apply transform to.

    Returns
    -------
    ds : xarray.Dataset
        Dataset after transform has been applied.
    """

    if len(ds.data_vars) == 0 or len(var.transform) == 0:
        logging.info("Dataset is empty, or no transforms -- skipping transform...")
        return ds

    if var.name not in ds:
        raise KeyError(f"Variable {var.name} not in dataset.")

    for transform in var.transform:
        try:
            x = 1.0  # noqa: F841
            eval(transform)  # x is implicitly a variable from the config
        except SyntaxError:
            raise SyntaxError(f"Invalid transform in config {transform}.")

        def func(x):
            return eval(transform)

        logging.info(f"Applying transform {transform} to {var.name}...")
        ds[var.name] = xr.apply_ufunc(func, ds[var.name], dask="parallelized")

    return ds


def compute_normalization(ds, varname, precomputed=None):
    if precomputed is None:
        logging.info("Computing min and max...")
        logging.info("Calculation min...")
        min = ds[varname].min().compute()
        logging.info("Calculation max...")
        max = ds[varname].max().compute()
    else:
        if (
            "min" not in precomputed[varname].attrs
            or "max" not in precomputed[varname].attrs
        ):
            raise KeyError(
                f"Precomputed dataset does not contain min and max for variable {varname}."
            )
        min = precomputed[varname].attrs["min"]
        max = precomputed[varname].attrs["max"]

    logging.info(f"Min: {min}, Max: {max}")

    if min == max:
        raise ZeroDivisionError("Min and max are equal.")

    if varname == "pr":
        eps = 10**-3
        ds[varname] = (np.log(ds[varname] + eps) - np.log(eps)) / (
            np.log(max + eps) - np.log(eps)
        )
    else:
        ds[varname] = (ds[varname] - min) / (max - min)

    ds[varname].attrs["min"] = float(min)
    ds[varname].attrs["max"] = float(max)

    return ds


def standardize(x: xr.DataArray, mean: float, std: float) -> xr.DataArray:
    """Standardize the data.

    Parameters
    ----------
    x : xarray.DataArray
        Data to standardize.
    mean : float
        Mean of the data.
    std : float
        Standard deviation of the data.

    Returns
    -------
    x : xarray.DataArray
        Standardized data.
    """
    if std == 0:
        raise ZeroDivisionError("Standard deviation is zero.")
    return (x - mean) / std


def compute_standardization(
    ds: xr.Dataset,
    varname: str,
    precomputed: xr.Dataset = None,
) -> xr.Dataset:  # sourcery skip: avoid-builtin-shadow
    """Standardize the statistics of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to standardize the statistics of.
    precomputed : xarray.Dataset, optional
        Dataset containing precomputed statistics, by default None
    Returns
    -------
    ds : xarray.Dataset
        Dataset with standardized statistics.
    """
    logging.info("Computing mean and standard deviation...")
    if precomputed is None:
        logging.info("Calculation mean...")
        mean = ds[varname].mean().compute()
        logging.info("Calculation std...")
        std = ds[varname].std().compute()
    else:
        if (
            "mean" not in precomputed[varname].attrs
            or "std" not in precomputed[varname].attrs
        ):
            raise KeyError(
                f"Precomputed dataset does not contain mean and std for variable {varname}."
            )
        mean = precomputed[varname].attrs["mean"]
        std = precomputed[varname].attrs["std"]

    ds[varname] = xr.apply_ufunc(
        standardize,
        ds[varname],
        mean,
        std,
        dask="parallelized",
    )

    ds[varname].attrs["mean"] = float(mean)
    ds[varname].attrs["std"] = float(std)

    return ds


# def split_and_standardize(ds, climdata, var) -> dict:
#     # Train test split
#     logging.info("Splitting dataset...")

#     ds = user_defined_transform(ds, var)

#     train_test = train_test_split(
#         ds, climdata.select.time.test_years, climdata.select.time.validation_years
#     )

#     # Standardize the dataset.
#     if var.apply_standardize:
#         logging.info(f"Standardizing {var.name}...")
#         train = compute_standardization(train_test["train"], var.name)
#         test = compute_standardization(
#             train_test["test"], var.name, train_test["train"]
#         )
#         validation = compute_standardization(
#             train_test["validation"], var.name, train_test["train"]
#         )

#     if var.apply_normalize:
#         logging.info(f"Normalizing {var.name}...")
#         train = compute_normalization(train_test["train"], var.name)
#         test = compute_normalization(train_test["test"], var.name, train_test["train"])
#         validation = compute_normalization(
#             train_test["validation"], var.name, train_test["train"]
#         )

#     if var.apply_normalize is False and var.apply_standardize is False:
#         logging.info("Skipping standardization and normalization...")
#         return {
#             "train": train_test["train"],
#             "test": train_test["test"],
#             "validation": train_test["validation"],
#         }

#     return {"train": train, "test": test, "validation": validation}


def standardize_or_normalize(ds: xr.Dataset, var: ClimateVariable) -> xr.Dataset:
    """
    Apply user-defined transformation, standardization, and normalization
    to a dataset for a given ClimateVariable.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the variable to transform.
    var : ClimateVariable
        The ClimateVariable object containing transformation and scaling flags.

    Returns
    -------
    xr.Dataset
        The dataset with the variable transformed, standardized, and/or normalized.

    Notes
    -----
    - Applies a user-defined transformation first (if any).
    - Standardization and normalization are applied in sequence.
    - If neither is requested, logs that the step was skipped.
    """
    ds = user_defined_transform(ds, var)

    if var.apply_standardize:
        logging.info(f"Standardizing {var.name}...")
        ds = compute_standardization(ds, var.name)

    elif var.apply_normalize:
        logging.info(f"Normalizing {var.name}...")
        ds = compute_normalization(ds, var.name)

    else:
        logging.info(f"Skipping standardization and normalization for {var.name}.")

    return ds
