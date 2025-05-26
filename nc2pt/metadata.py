import logging
from hydra.utils import instantiate
from typing import Union, List
from nc2pt.climatedata import ClimateDimension, ClimateVariable, ClimateData
import xarray as xr


class MultipleKeys(Exception):
    """Raised when a variable has multiple keys in the dataset."""

    pass


class MissingKey(Exception):
    """Raised when a variable is missing from the dataset."""

    pass


def configure_metadata(
    ds: xr.Dataset, var: ClimateVariable, climdata: ClimateData
) -> xr.Dataset:
    """Function to be called before preprocessing.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to preprocess.

    Returns
    -------
    ds : xarray.Dataset
        Dataset after preprocessing.
    """

    logging.info("✨ Homogenizing dataset keys...")

    dim_coord_attrs = {"coords": climdata.coords, "dims": climdata.dims}

    if "forecast_initial_time" in ds.dims and "forecast_hour" in ds.dims:
        # Flatten 2D time coordinates found in ERA5 PR nc files
        ds = flatten_2D_forecast_time(ds=ds)

    for ds_attr, dim_or_coord in dim_coord_attrs.items():
        ds = loop_over_keys_and_rename(ds, dim_or_coord, ds_attr)

    if var.name != "hr_ref" and len(climdata.dims) != len(getattr(ds, "dims")):
        # Remove the keys that aren't in the climatedata dims.
        keys = getattr(ds, "dims")

        dim_or_coord_attrs = {
            dim_or_coord.name for dim_or_coord in climdata.dims + climdata.coords
        }
        ds_attrs = set(keys.keys())

        keys_to_remove = list(ds_attrs - dim_or_coord_attrs)
        logging.info(f"Dropping {keys_to_remove} from dataset.")

        # assert 0

        for k in keys_to_remove:
            if ds[k].size == 1:
                ds = ds.squeeze(k).drop_vars(k)

    ds = rename_keys(ds, outcome_obj=var, ds_attr="data_vars")
    preserve = [dim.name for dim in climdata.dims] + [coord.name for coord in climdata.coords]
    ds = drop_unused_variables(ds, var, preserve_vars=preserve)
    ds = match_longitudes(ds) if var.is_west_negative else ds

    return ds


def loop_over_keys_and_rename(ds, dim_or_coord, ds_attr):
    for x in dim_or_coord:
        x = instantiate(x)
        ds = rename_keys(ds, outcome_obj=x, ds_attr=ds_attr)

    return ds


def rename_keys(
    ds: xr.Dataset,
    outcome_obj: Union[ClimateDimension, ClimateVariable],
    ds_attr: str,
) -> xr.Dataset:
    """
    Renames variables in a dataset based on alternative names
    provided for a ClimateVariable or ClimateDimension.

    This function takes a dataset, a ClimateVariable or ClimateDimension object,
    and a dataset attribute name.
    It checks if any of the alternative names for the ClimateVariable or ClimateDimension
    match keys in the dataset attribute.
    If there is exactly one match, it renames that key to the standard name in the ClimateVariable
    or ClimateDimension.

    Args:
    ds: xarray Dataset to rename
    outcome_obj: The ClimateVariable or ClimateDimension
    ds_attr: Attribute of the dataset to check for keys (e.g. 'variables' or 'coords')

    Returns:
    xr.Dataset: Dataset with renamed variables

    Raises:
    MultipleKeys: If there are multiple matching keys
    """

    keys = getattr(ds, ds_attr)
    keymatch = [i for i in outcome_obj.alternative_names if i in keys]
    # Rename the variable if it is listed as an alternative name.
    if len(keymatch) == 1:
        old_name = keymatch[0]
        ds = ds.rename({old_name: outcome_obj.name})
        logging.info(f"Renamed {old_name} to {outcome_obj.name}")
    elif len(keymatch) > 1:
        raise MultipleKeys(f"{outcome_obj.name} has multiple alternatives in dataset.")

    return ds


def match_longitudes(ds: xr.Dataset) -> xr.Dataset:
    """Match the longitudes of the dataset to the range [-180, 180].

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to match the longitudes of.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with longitudes in the range [-180, 180].
    """
    if ds.lon.min() > 0:
        raise ValueError(
            "Dataset longitudes are likely not in the range [-180, 180]"
            "which is the intention of this function. Check longitude units."
        )
    ds = ds.assign_coords(lon=(ds.lon + 360))
    return ds


def flatten_2D_forecast_time(ds: xr.Dataset) -> xr.Dataset:
    """Flatten forecast_initial_time and forecast_hour into a
      single time coordinate (seems to be structure of ERA5 PR data).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing 'forecast_initial_time' and 'forecast_hour'.

    Returns
    -------
    ds : xarray.Dataset
        Dataset with stacked 'time' dimension based on valid forecast time.
    """
    init, hour = xr.broadcast(ds.forecast_initial_time, ds.forecast_hour)
    valid_time = init + hour.astype("timedelta64[ns]")
    ds = ds.stack(time=("forecast_initial_time", "forecast_hour"))
    ds = ds.drop_vars(["time", "forecast_initial_time", "forecast_hour"])
    ds = ds.assign_coords(time=("time", valid_time.values.ravel()))
    ds = ds.assign_coords(time=("time", valid_time.values.ravel()))
    return ds


def drop_unused_variables(
    ds: xr.Dataset,
    var: ClimateVariable,
    preserve_vars: List[str]
) -> xr.Dataset:
    """
    Drop all data variables from the dataset except the one declared in the ClimateVariable object
    and any variables explicitly listed to be preserved (e.g., dimensions and coordinates).

    Parameters
    ----------
    ds : xr.Dataset
        The dataset potentially containing multiple variables.
    var : ClimateVariable
        The ClimateVariable object specifying the intended variable to keep.
    preserve_vars : List[str]
        A list of variable names to retain in addition to the main variable (e.g., time, lat, lon).

    Returns
    -------
    xr.Dataset
        A filtered dataset containing only the target variable and the preserved variables.

    Logs
    ----
    Logs a message listing all dropped variables, if any.

    Raises
    ------
    KeyError
        If the target variable is not found in the dataset.
    """
    if var.name not in ds:
        raise KeyError(f"Declared variable '{var.name}' not found in dataset variables: {list(ds.variables)}")

    keep_vars = {var.name, *preserve_vars}
    all_vars = set(ds.variables)
    to_drop = all_vars - keep_vars

    if to_drop:
        logging.info(f"Dropping unused variables from {var.path}: {sorted(to_drop)}")

    ds = ds.drop_vars(to_drop, errors="ignore")

    return ds