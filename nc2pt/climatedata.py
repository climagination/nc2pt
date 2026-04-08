from dataclasses import dataclass, field
import logging
from typing import Optional, List, Union, Any, Dict
from pathlib import Path
import xarray as xr
import json


@dataclass
class ClimateDimension:
    name: str
    alternative_names: List[str]
    hr_only: Optional[bool] = field(default=False)
    chunksize: Union[int, str] = field(default="auto")


@dataclass
class ClimateVariable:
    name: str
    alternative_names: List[str]
    path: str
    is_west_negative: bool
    transform: Optional[str] = field(default=None)
    invariant: Optional[bool] = field(default=False)
    apply_standardize: Optional[bool] = field(default=True)
    apply_normalize: Optional[bool] = field(default=False)
    metadata_path: Optional[str] = field(default=None)
    coarsening_method: Optional[str] = field(default='mean')


# Write a dataclass that loads config data from hydra-core and
# populates the class with the config data.
@dataclass
class ClimateModel:
    # These will come from instantiating the class with hydra.
    name: str
    info: str
    climate_variables: List[ClimateVariable]
    hr_ref: Optional[ClimateVariable] = None
    engine: Optional[str] = None  # Optional engine override for this model
    emulation_data: Optional[bool] = False  # Optional bool to handle metadata ingestion
    loader: Optional[str] = "default" # Optional flag for UBC WRF specific io 'ubc_wrf'
    alignment_pipeline: List[str] = field(default_factory=lambda: [
        "temporal_crop", "regrid", "spatial_crop", "coarsen", "user_defined_transforms", "split_data"
    ])


@dataclass
class ClimateData:
    output_path: str
    climate_models: List[ClimateModel]
    dims: List[ClimateDimension]
    coords: List[ClimateDimension]
    select: dict
    compute: dict
    loader: dict
    internal: Dict[str, Any] = field(default_factory=dict, repr=False)

    def apply_chunks(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Apply dimension-aware chunking to an xarray Dataset using the ClimateDimension specs.

        Only includes dimensions that exist in the dataset. Falls back to alternative names.
        """
        chunk_dict = {}

        for dim in self.dims:
            # Check if dim.name or any alternative name is present in the dataset
            matched_name = next(
                (name for name in [dim.name] + dim.alternative_names if name in ds.dims),
                None
            )
            if matched_name is not None:
                chunk_dict[matched_name] = dim.chunksize

        if chunk_dict:
            logging.info(f"Applying chunks: {chunk_dict}")
            return ds.chunk(chunk_dict)
        else:
            logging.warning("No matching dimensions found for chunking.")
            return ds

    def __post_init__(self):
        for model in self.climate_models:
            logging.info(f"🌎 Instantiated Model with information: {model.info}")


@dataclass
class FeatureScalingMetadata:
    variable: str
    method: str  # "minmax" or "standard"
    min: Optional[float] = None
    max: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    apply_normalize: bool = False
    apply_standardize: bool = False
    transforms: List = False
    is_west_negative: bool = False
    spatial_scale_factor: int = 1
    spatial_crop: Optional[dict] = None
    hr_reference_field: Optional[str] = None
    units: Optional[dict] = None
    created: Optional[str] = None
    hash: Optional[str] = None

    @classmethod
    def from_json(cls, path: Path) -> "FeatureScalingMetadata":
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)

    def get_stats(self) -> dict[str, float]:
        if self.apply_normalize and self.method == "minmax":
            if self.min is None or self.max is None:
                raise ValueError("Min-max normalization requires 'min' and 'max'")
            return {"min": self.min, "max": self.max}
        elif self.apply_standardize and self.method == "standard":
            if self.mean is None or self.std is None:
                raise ValueError("Standardization requires 'mean' and 'std'")
            return {"mean": self.mean, "std": self.std}
        else:
            raise ValueError(
                f"Inconsistent method or missing stats in metadata for variable '{self.variable}'"
            )

    def validate_against_var(self, var: "ClimateVariable"):
        if self.variable != var.name:
            raise ValueError(f"Variable name mismatch: {self.variable} != {var.name}")
        if self.apply_normalize != var.apply_normalize:
            raise ValueError("Mismatch in 'apply_normalize' between var config and metadata.")
        if self.apply_standardize != var.apply_standardize:
            raise ValueError("Mismatch in 'apply_standardize' between var config and metadata.")
