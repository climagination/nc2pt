from datetime import datetime
import xarray as xr


def load_grid(path: str, engine: str = "netcdf4", chunks: int = 250) -> xr.Dataset:
    """Load the grid to regrid to.

    Parameters
    ----------
    path : str
        Path to the grid to regrid to.

    Returns
    -------
    grid : xarray.Dataset
        Grid to regrid to.
    """

    if "*" in path or isinstance(path, list):
        with xr.open_mfdataset(path, engine=engine, parallel=True, chunks="auto") as ds:
            return ds
    else:
        with xr.open_dataset(path, engine=engine, chunks="auto") as ds:
            return ds


def write_to_zarr(ds: xr.Dataset, path: str) -> None:
    """Write the output to disk.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to write to disk.
    path : str
        Path to write the dataset to.
    """

    # See https://github.com/pydata/xarray/issues/5219
    ds = ds.assign_attrs(
        {
            "history": f"Created by {__file__} on {datetime.now()}",
        }
    )
    ds.to_zarr(f"{path}.zarr", mode="w", consolidated=True)
