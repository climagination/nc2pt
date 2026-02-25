from nc2pt.metadata import configure_metadata, NormalizerMetadataCollector
from nc2pt.io import load_grid, write_to_zarr
from nc2pt.align import run_alignment_pipeline
from nc2pt.scaling import apply_feature_scaling
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
    metadata_collector = NormalizerMetadataCollector(climdata.output_path)
    if model.hr_ref is not None:
        hr_ref = load_grid(model.hr_ref.path, engine=climdata.compute.engine)
        logging.info("👀 Processing high resolution reference field...")
        hr_ref = configure_metadata_fn(hr_ref, instantiate(model.hr_ref))
        metadata_collector.save_grid_if_needed(hr_ref, model.name, climdata)
    else: hr_ref = None

    for climate_variable in model.climate_variables:
        # Instantiates climate_variable object in cliamtedata.py
        climate_variable = instantiate(climate_variable)
        engine = model.engine or climdata.compute.engine
        loader = model.loader

        logging.info(f"Loading {climate_variable.name} data")
        ds = load_grid(climate_variable.path, engine=engine, loader=loader)

        start = timer()
        logging.info(
            f"✨ Starting {climate_variable.name} from {model.info} input dataset..."
        )

        ds = configure_metadata_fn(ds, climate_variable)
        metadata_collector.log_variable_units_from_dataset(climate_variable.name, ds)
        ds = climdata.apply_chunks(ds)
        ds_aligned = run_alignment_pipeline(ds, climate_variable, model, climdata, hr_ref, metadata_collector)
        ds_aligned = apply_feature_scaling(ds_aligned, climate_variable, model)
        reference_key = "train" if "train" in ds_aligned else "full"

        metadata_collector.add_variable_from_config(
            climate_data=climdata,
            climate_variable=climate_variable,
            processed_ds=ds_aligned[reference_key],
            model_name=model.name
        )

        for dataset_key, dataset in ds_aligned.items():
            # Skip 'full' in filename if not split
            if dataset_key == "full":
                out_path = f"{climdata.output_path}/{climate_variable.name}_{model.name}"
            else:
                out_path = f"{climdata.output_path}/{climate_variable.name}_{dataset_key}_{model.name}"
            logging.info(f"✨ Writing {dataset_key} output...")
            write_to_zarr(climdata.apply_chunks(dataset), out_path)

        end = timer()
        logging.info(f"🎉 Done processing {climate_variable.name} in {model.info}!")
        logging.info(f"⏳ Time elapsed ⏳: {timedelta(seconds=end-start)}")
        del ds, ds_aligned

    metadata_collector.write_all(model_name=model.name)


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
