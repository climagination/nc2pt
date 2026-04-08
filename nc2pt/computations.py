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
            x = 1.0
            eval(transform, {"np": np, "x": x})  # x is implicitly a variable from the config
        except SyntaxError:
            raise SyntaxError(f"Invalid transform in config {transform}.")

        def func(x):
            return eval(transform, {"np": np, "x": x})

        logging.info(f"🧮 Applying transform {transform} to {var.name}...")
        ds[var.name] = xr.apply_ufunc(func, ds[var.name], dask="parallelized")

    return ds


def compute_normalization(ds, varname, precomputed=None, feature_scaling_stats=None):
    """ Normalize the statistics of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to standardize the statistics of.
    varname: str
        String containing variable name
    feature_scaling_stat: dict
        Dictionary containing max and min, by default None
    precomputed : xarray.Dataset, optional
        Dataset containing precomputed statistics, by default None
    Returns
    -------
    ds : xarray.Dataset
        Dataset with standardized statistics.
    """
    if (precomputed is None and (not feature_scaling_stats)):
        logging.info("Computing min and max...")
        logging.info("Calculation min...")
        min = ds[varname].min().compute()
        logging.info("Calculation max...")
        max = ds[varname].max().compute()
    elif feature_scaling_stats is not None:
        min = feature_scaling_stats['min']
        max = feature_scaling_stats['max']
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

    # if varname == "pr":
    #     eps = 10**-3
    #     ds[varname] = (np.log(ds[varname] + eps) - np.log(eps)) / (
    #         np.log(max + eps) - np.log(eps)
    #     )
    #     logging.info(
    #     "Applied log-transform + min-max scaling to precipitation ('pr'). See README for more details."
    # )
    # else:
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
    feature_scaling_stats: dict[str, float] = None,
    precomputed: xr.Dataset = None,
) -> xr.Dataset:  # sourcery skip: avoid-builtin-shadow
    """Standardize the statistics of the dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to standardize the statistics of.
    varname: str
        String containing variable name
    feature_scaling_stat: dict
        Dictionary containing mean and std, by default None
    precomputed : xarray.Dataset, optional
        Dataset containing precomputed statistics, by default None
    Returns
    -------
    ds : xarray.Dataset
        Dataset with standardized statistics.
    """

    logging.info("Computing mean and standard deviation...")

    if precomputed is None and feature_scaling_stats is None:
        logging.info("Calculation mean...")
        mean = ds[varname].mean().compute()
        logging.info("Calculation std...")
        std = ds[varname].std().compute()
    elif feature_scaling_stats is not None:
        mean = feature_scaling_stats['mean']
        std = feature_scaling_stats['std']
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
