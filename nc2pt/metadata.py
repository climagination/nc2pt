import json
import hashlib
from datetime import datetime
from pathlib import Path
import logging
from hydra.utils import instantiate
from typing import Union, List
from nc2pt.climatedata import ClimateDimension, ClimateVariable, ClimateData
import xarray as xr
from omegaconf import DictConfig, ListConfig, OmegaConf


class MultipleKeys(Exception):
    """Raised when a variable has multiple keys in the dataset."""

    pass


class MissingKey(Exception):
    """Raised when a variable is missing from the dataset."""

    pass


class NormalizerMetadataCollector:
    """
    Collects and writes normalization or standardization metadata for climate variables.

    Tracks attributes like min/max or mean/std, units, transforms, and spatial settings
    for each variable after standardization/normalization is applied. Saves the metadata
    to per-variable JSON files with a content hash for reproducibility and validation.
    """

    def __init__(self, output_dir: str):
        """
        Initialize the collector and create the output directory.

        Args:
            output_dir: Directory where normalization JSON files will be written.
        """
        self.metadata = {}
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"📁 NormalizerMetadataCollector initialized with output dir: {self.output_dir}")

    def _compute_hash(self, data: dict) -> str:
        """
        Compute a SHA256 hash of the metadata dictionary (excluding the 'hash' field).

        Args:
            data: Dictionary to hash.

        Returns:
            SHA256 hash string.
        """
        # Exclude 'hash' and convert each value if needed
        temp = {}
        for k, v in data.items():
            if k == "hash":
                continue
            if OmegaConf.is_config(v):  # Handles both DictConfig and ListConfig
                temp[k] = OmegaConf.to_container(v, resolve=True)
            else:
                temp[k] = v

        temp_json = json.dumps(temp, sort_keys=True)
        return hashlib.sha256(temp_json.encode()).hexdigest()

    def _extract_stats(self, variable: ClimateVariable, attrs: dict) -> dict:
        """
        Extract normalization or standardization statistics from xarray attributes.

        Args:
            variable: The ClimateVariable being processed.
            attrs: The attributes from the xarray variable.

        Returns:
            A dictionary containing method and statistics (min/max or mean/std).
        """
        if variable.apply_normalize:
            return {
                "method": "minmax",
                "min": float(attrs["min"]),
                "max": float(attrs["max"]),
            }
        elif variable.apply_standardize:
            return {
                "method": "standard",
                "mean": float(attrs["mean"]),
                "std": float(attrs["std"]),
            }
        return {}

    def _get_spatial_crop(self, climate_data: ClimateData) -> dict:
        """
        Retrieve the spatial crop bounds from the config.

        Args:
            climate_data: Full ClimateData config object.

        Returns:
            Dict with x and y crop bounds.
        """
        spatial_cfg = climate_data.select["spatial"]
        return {
            "x": [spatial_cfg["x"]["first_index"], spatial_cfg["x"]["last_index"]],
            "y": [spatial_cfg["y"]["first_index"], spatial_cfg["y"]["last_index"]],
        }

    def _get_hr_ref_path(self, climate_data: ClimateData, model_name: str) -> str:
        """
        Look up the HR reference field path from the model config.

        Args:
            climate_data: Full ClimateData config.
            model_name: Name of the model (e.g., 'lr').

        Returns:
            The path to the HR reference field, or None.
        """
        for model in climate_data.climate_models:
            if model.name == model_name and model.hr_ref is not None:
                return model.hr_ref.path
        return None
    
    def _convert_nested_omegaconf(self, obj):
        """
        Recursively convert any OmegaConf DictConfig or ListConfig to plain Python dicts/lists.

        Parameters
        ----------
        obj : Any
            Object potentially containing OmegaConf configs.

        Returns
        -------
        Any
            A fully OmegaConf-free version of the input object.
        """
        if isinstance(obj, (DictConfig, ListConfig)):
            return OmegaConf.to_container(obj, resolve=True)
        elif isinstance(obj, dict):
            return {k: self._convert_nested_omegaconf(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_nested_omegaconf(v) for v in obj]
        else:
            return obj

    def log_variable_units_from_dataset(self, variable_name: str, ds: xr.Dataset) -> None:
        """
        Extract and log the original units for a given variable from a dataset.

        This should be called early in the workflow, before normalization or 
        standardization is applied. It stores the 'original' units in the metadata, 
        and leaves 'post_transform' as 'unknown' for now.

        Args:
            variable_name (str): Name of the climate variable (e.g., 'uas').
            ds (xarray.Dataset): Dataset containing the variable.
        """
        units = ds[variable_name].attrs.get("units", "unknown")
        logging.info(f"Extracted units for variable '{variable_name}': {units}")

        if variable_name not in self.metadata:
            self.metadata[variable_name] = {
                "units": {"original": units, "post_transform": "unknown"}
            }
        else:
            self.metadata[variable_name].setdefault("units", {})
            self.metadata[variable_name]["units"]["original"] = units
            self.metadata[variable_name]["units"].setdefault("post_transform", "unknown")

    def add_variable_from_config(
        self,
        climate_data: ClimateData,
        climate_variable: ClimateVariable,
        processed_ds: xr.Dataset,
        model_name: str,
    ):
        """
        Add metadata for a single climate variable after it's been normalized or standardized.

        Args:
            climate_data: Full ClimateData config object.
            climate_variable: The ClimateVariable being processed.
            processed_ds: xarray.Dataset that contains processed data for this variable.
            model_name: Name of the model this variable belongs to (e.g., 'lr').
        """
        var_name = climate_variable.name
        attrs = processed_ds[var_name].attrs

        stats = self._extract_stats(climate_variable, attrs)
        if not stats:
            logging.info(f"⚠️  Skipping variable '{var_name}': not scaled.")
            return

        logging.info(f"📦 Adding metadata for variable: {var_name}")

        self.metadata.setdefault(var_name, {})  # Preserve existing entries like "units"

        self.metadata[var_name].update({
            "variable": var_name,
            **stats,
            "apply_normalize": climate_variable.apply_normalize,
            "apply_standardize": climate_variable.apply_standardize,
            "transforms": climate_variable.transform or [],
            "is_west_negative": climate_variable.is_west_negative,
            "spatial_scale_factor": climate_data.select["spatial"]["scale_factor"],
            "spatial_crop": self._get_spatial_crop(climate_data),
            "hr_reference_field": self._get_hr_ref_path(climate_data, model_name),
            "created": datetime.utcnow().isoformat() + "Z",
        })

        # Ensure 'units' dict exists and has both keys set
        self.metadata[var_name].setdefault("units", {})
        self.metadata[var_name]["units"].setdefault("original", "unknown")
        self.metadata[var_name]["units"].setdefault("post_transform", "unknown")

    def write_all(self, model_name: str):
        """
        Write each variable's metadata to a JSON file in the output directory.
        Adds a hash field for traceability.
        """
        if not self.metadata:
            logging.warning("⚠️ No metadata to write.")
            return

        for var_name, meta in self.metadata.items():
            # Add the hash first (must be added to the original dict)
            meta["hash"] = self._compute_hash(meta)

            # Convert entire metadata dict to a plain dict recursively
            meta_clean = self._convert_nested_omegaconf(meta)

            out_path = self.output_dir / f"{model_name}_{var_name}_feature_scaling_metadata.json"
            logging.info(f"📝 Writing metadata for '{var_name}' to {out_path}")
            with out_path.open("w") as f:
                json.dump(meta_clean, f, indent=2)
            logging.info(f"✅ Wrote: {out_path}")


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
    valid_time = init + hour.astype("timedelta64[h]")

    ds = ds.stack(time=("forecast_initial_time", "forecast_hour"))
    ds = ds.reset_index("time", drop=True)
    ds["time"] = valid_time.stack(time=("forecast_initial_time", "forecast_hour")).values
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
