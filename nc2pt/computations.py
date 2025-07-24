import logging
import xarray as xr
import numpy as np  # noqa: F401s
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

        logging.info(f"🧮 Applying transform {transform} to {var.name}...")
        ds[var.name] = xr.apply_ufunc(func, ds[var.name], dask="parallelized")

    return ds


def compute_normalization(ds, varname, precomputed=None):
    logging.info(f"Normalizing {varname}...")
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
    logging.info(f"Standardizing {varname}...")
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
