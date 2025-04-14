from nc2pt.metadata import configure_metadata
from nc2pt.io import load_grid, write_to_zarr
from nc2pt.align import align_with_lr, crop_field, slice_time
from nc2pt.computations import split_and_standardize, user_defined_transform
from nc2pt.climatedata import ClimateData, ClimateModel

import logging
from datetime import timedelta
from functools import partial
from timeit import default_timer as timer
import hydra
from hydra.utils import instantiate
from dask.distributed import Client
import dask


def preprocess_variables(model: ClimateModel, climdata: ClimateData) -> None:
    """Preprocesses climate variables from model data.

    Loads configured climate variables from file, aligns high/low resolution grids,
    applies transforms, splits into train/test sets, standardizes, and writes output.

    Args:
    model: Model configuration.
    climdata: Climate data configuration.

    Returns:
    None
    """

    configure_metadata_fn = partial(configure_metadata, climdata=climdata)
    alignment_procedures = {}
    if model.hr_ref is not None:
        hr_ref = load_grid(model.hr_ref.path, engine=climdata.compute.engine)
        logging.info("👀 Processing high resolution reference field...")
        hr_ref = configure_metadata_fn(hr_ref, instantiate(model.hr_ref))
        alignment_procedures |= {
            "lr": partial(align_with_lr, hr_ref=hr_ref, climdata=climdata),
            "lr_invariant": partial(align_with_lr, hr_ref=hr_ref, climdata=climdata),
        }

    for climate_variable in model.climate_variables:
        # Instantiates climate_variable object in cliamtedata.py
        climate_variable = instantiate(climate_variable)
        chunk_dims = {dim.name: dim.chunksize for dim in climdata.dims}
        chunk_coords = {coord.name: coord.chunksize for coord in climdata.coords}
        ds = load_grid(climate_variable.path, engine=climdata.compute.engine)

        start = timer()
        logging.info(
            f"✨ Starting {climate_variable.name} from {model.info} input dataset..."
        )

        ds = configure_metadata_fn(ds, climate_variable)
        if 'rlat' in ds.dims:
            ds = ds.chunk(chunk_dims)
        elif 'lat' in ds.dims:
            ds = ds.chunk(chunk_coords)
        else:
            raise ValueError("Dataset must have either 'lat' or 'rlat' in its dimensions.")

        ds = (
            slice_time(
                ds, climdata.select.time.range.start, climdata.select.time.range.end
            )
            if climate_variable.invariant is False
            else ds
        )

        alignment_procedures |= {
            "hr": partial(
                crop_field,
                scale_factor=climdata.select.spatial.scale_factor,
                x=climdata.select.spatial.x,
                y=climdata.select.spatial.y,
            ),
            "hr_invariant": partial(
                crop_field,
                scale_factor=climdata.select.spatial.scale_factor,
                x=climdata.select.spatial.x,
                y=climdata.select.spatial.y,
            ),
        }

        # This implies that it is a different grid or a lr dataset.
        ds = alignment_procedures[model.name](ds)
        if climate_variable.invariant is False:
            train_test_validation_ds = split_and_standardize(
                ds, climdata, climate_variable
            )
            for train_test_validation in train_test_validation_ds:
                logging.info(f"✨ Writing {train_test_validation} output...")
                write_to_zarr(
                    train_test_validation_ds[train_test_validation].chunk(chunk_dims),
                    f"{climdata.output_path}/{climate_variable.name}_{train_test_validation}_{model.name}",
                )

        else:
            logging.info("✨ Writing output...")
            write_to_zarr(
                ds.chunk(chunk_dims),
                f"{climdata.output_path}/{climate_variable.name}_{model.name}",
            )

        end = timer()
        logging.info(f"🎉 Done processing {climate_variable.name} in {model.info}!")
        logging.info(f"⏳ Time elapsed ⏳: {timedelta(seconds=end-start)}")
        del ds


def preprocess(climdata: ClimateData) -> None:
    for climate_model in climdata.climate_models:
        climate_model = instantiate(climate_model)
        preprocess_variables(climate_model, climdata)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def start(climate_data) -> None:
    with Client(
        processes=True,
        dashboard_address=climate_data.compute.dask_dashboard_address,
    ):
        climate_data = instantiate(climate_data)
        preprocess(climate_data)


if __name__ == "__main__":
    with dask.config.set(**{"array.slicing.split_large_chunks": False}):
        start()
