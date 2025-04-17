from dataclasses import dataclass, field
import logging
from typing import Optional, List, Union
import xarray as xr


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


# Write a dataclass that loads config data from hydra-core and
# populates the class with the config data.
@dataclass
class ClimateModel:
    # These will come from instantiating the class with hydra.
    name: str
    info: str
    climate_variables: List[ClimateVariable]
    hr_ref: Optional[ClimateVariable] = None

    def __post_init__(self):
        logging.info(f"🌎 Instantiated Model with information: {self.info}")


@dataclass
class ClimateData:
    output_path: str
    climate_models: List[ClimateModel]
    dims: List[ClimateDimension]
    coords: List[ClimateDimension]
    select: dict
    compute: dict
    loader: dict

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