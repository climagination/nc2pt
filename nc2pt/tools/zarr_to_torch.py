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

    with xr.open_zarr(
        f"{output_path}/{var.name}_{s}_{model.name}.zarr/", chunks=chunks
    ) as ds:
        ds = ds.sortby("time")
        # Create parent dir if it doesn't exist for each variable
        make_dirs(output_path, s, var.name, model.name)
        indices = ds.time.dt.strftime("%Y-%m-%d-%H").values

        partial_paths = [
            f"{output_path}/{s}/{var.name}/{model.name}/{var.name}_{i}.pt"
            for i in indices
        ]

        pool_tuple = zip(
            indices,
            partial_paths,
            ds[var.name].transpose(*chunks.keys()),
        )

        progress_starmap(
            parallel_loop, pool_tuple, total=ds.time.size, n_cpu=24, chunk_size=1
        )


def loop_over_sets(climate_data, model, s):
    model = instantiate(model)

    logging.info(f"Loading {s} {model.name} dataset...")
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

        partial_set_loop = partial(loop_over_sets, climate_data, model)
        for s in ["train", "test", "validation"]:
            partial_set_loop(s)

        end = timer()
        logging.info(f"Finished {model.name} dataset in {timedelta(seconds=end-start)}")


if __name__ == "__main__":
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    main()
