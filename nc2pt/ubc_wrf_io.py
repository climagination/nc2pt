"""UBC WRF-specific data loading utilities."""

import xarray as xr
import pandas as pd
import re
from pathlib import Path


def get_ubc_wrf_file_list(pattern: str) -> list:
    """Get file list"""
    import glob

    print("Scanning for files (this may take ~30s over NFS)...")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(files)} files")

    return files


def add_ubc_wrf_timesteps(ds):
    """
    Preprocess WRF metgrid files with dummy Time coordinate.
    Extracts date from filename and creates proper time axis.
    Drops last timestep (corresponding to next month's 00:00:00).
    """
    # Get filename from dataset encoding
    filepath = ds.encoding.get('source', '')
    
    # Extract year and month: metgrid_YYYY_MM.nc
    match = re.search(r'metgrid_(\d{4})_(\d{2})\.nc$', filepath)
    
    if match:
        year, month = match.groups()
        start_date = f"{year}-{month}-01"
        
        # Drop last timestep (spin-up for next month)
        ds = ds.isel(Times=slice(None, -1))
        
        # Create time coordinates for remaining timesteps
        n_times = ds.sizes['Times']
        time_coords = pd.date_range(start=start_date, periods=n_times, freq='h')
        ds = ds.assign_coords(Times=time_coords)
    else:
        # For invariant fields or unparseable files, drop Time if it's dummy
        if 'Times' in ds.dims and len(ds.Times) <= 2:
            ds = ds.isel(Times=0, drop=True)
    
    return ds


def load_ubc_wrf(path: str, engine: str = "netcdf4", chunks: str = "auto") -> xr.Dataset:
    """Load WRF metgrid files with proper time coordinate handling."""
    
    if "*" in path or isinstance(path, list):
        if isinstance(path, str):
            file_list = get_ubc_wrf_file_list(path)
        else:
            file_list = path
        
        print(f"Opening {len(file_list)} files...")
        print("Note: Dropping last timestep of each month (corresponds to M+1 00:00:00)")  # Log once here
        
        ds = xr.open_mfdataset(
            file_list,
            engine=engine,
            parallel=True,
            chunks='auto',
            preprocess=add_ubc_wrf_timesteps,
            combine='nested',
            concat_dim='Times',
            combine_attrs='override',
            data_vars='minimal',
            coords='minimal',
            compat='override'
        )
        return ds
    else:
        ds = xr.open_dataset(path, engine=engine, chunks=chunks)
        return add_ubc_wrf_timesteps(ds)