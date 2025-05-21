import os
import logging
from datetime import timedelta
from timeit import default_timer as timer
from functools import partial
from pathlib import Path

import xarray as xr
import torch
import numpy as np
from hydra.utils import instantiate
import hydra

from parallelbar import progress_starmap


def parallel_loop(i: int, path: str, arr: xr.DataArray):
    # i, path, arr = tup
    # get yyyy-mm-dd frmo arr and add it to filename
    arr = arr.values
    x = torch.tensor(np.array(arr))
    assert not torch.isnan(x).any(), f"NaNs found in {i}"
    torch.save(x, path)


def make_dirs(output_path: str, s, var_name: str, model_name: str) -> None:
    if not os.path.exists(f"{output_path}/{s}/{var_name}/{model_name}"):
        os.makedirs(f"{output_path}/{s}/{var_name}/{model_name}")


def loop_over_variables(climate_data, model, var, s):
    climate_data = instantiate(climate_data)
    output_path = climate_data.output_path
    dims = [instantiate(dim) for dim in climate_data.dims]
    chunks = {dim.name: dim.chunksize for dim in dims}

    output_subfolder = "invariant" if s is None else s

    if s is None:
        zarr_path = f"{output_path}/{var.name}_{model.name}.zarr/"
    else:
        zarr_path = f"{output_path}/{var.name}_{s}_{model.name}.zarr/"

    with xr.open_zarr(zarr_path, chunks=chunks) as ds:

        # Handle invariant variable
        if "time" not in ds.dims or getattr(var, "invariant", False):
            logging.info(f"{var.name} is invariant. Saving without time slicing.")
            make_dirs(output_path, output_subfolder, var.name, model.name)

            path = f"{output_path}/{output_subfolder}/{var.name}/{model.name}/{var.name}.pt"
            arr = ds[var.name].values
            x = torch.tensor(np.array(arr))
            assert not torch.isnan(x).any(), f"NaNs found in {var.name}"
            torch.save(x, path)

            # Save attributes
            attrs = ds[var.name].attrs
            attrs_file = Path(
                f"{output_path}/{output_subfolder}/{var.name}/{model.name}/{var.name}_attrs.yaml"
            )
            with open(attrs_file, "w") as f:
                f.write("attributes:\n")
                for key, value in attrs.items():
                    f.write(f"  {key}: {value}\n")
            return

        ds = ds.sortby("time")
        # Create parent dir if it doesn't exist for each variable
        make_dirs(output_path, output_subfolder, var.name, model.name)
        indices = ds.time.dt.strftime("%Y-%m-%d-%H").values
        
        partial_paths = [
            f"{output_path}/{output_subfolder}/{var.name}/{model.name}/{var.name}_{i}.pt"
            for i in indices
        ]

        pool_tuple = zip(
            indices,
            partial_paths,
            ds[var.name].transpose(*[dim for dim in chunks.keys() if dim in ds[var.name].dims]),
        )

        progress_starmap(
            parallel_loop, pool_tuple, total=ds.time.size, n_cpu=24, chunk_size=1
        )


def loop_over_sets(climate_data, model, s: str = None):
    model = instantiate(model)
    label = s if s else "invariant"

    logging.info(f"Loading {label} {model.name} dataset...")
    for var in model.climate_variables:
        logging.info(f"Processing {var.name} variable...")
        if s == "train":
            # load train file and write the attributes to a yaml file
            train_file = f"{climate_data.output_path}/{var.name}_{s}_{model.name}.zarr"
            train_ds = xr.open_zarr(train_file)
            train_attrs = train_ds[var.name].attrs
            # now write the attributes to a yaml file
            train_dir = Path(f"{climate_data.output_path}/{s}")
            train_dir.mkdir(parents=True, exist_ok=True)

            train_attrs_file = train_dir / f"{var.name}_{s}_{model.name}_attrs.yaml"
            with open(train_attrs_file, "w") as file:
                file.write("attributes:\n")
                for key, value in train_attrs.items():
                    file.write(f"  {key}: {value}\n")

        loop_over_variables(climate_data, model, var, s)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(climate_data) -> None:
    # Define for loop that iterates over sets, resolutions, and variables
    # and saves each time step as a torch tensor to write to a pytorch file
    # format.
    climate_data = instantiate(climate_data)
    for model in climate_data.climate_models:
        start = timer()

        if all(getattr(var, "invariant", False) for var in model.climate_variables):
            logging.info(f"Processing invariant model: {model.name}")
            loop_over_sets(climate_data, model, s=None)
        else:
            partial_set_loop = partial(loop_over_sets, climate_data, model)
            for s in ["train", "test", "validation"]:
                partial_set_loop(s)

        end = timer()
        logging.info(f"Finished {model.name} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
