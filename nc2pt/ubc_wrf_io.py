"""UBC WRF-specific data loading utilities."""

import xarray as xr
import pandas as pd
import re
import pickle
from pathlib import Path


def get_ubc_wrf_file_list(pattern: str, cache_file: str = None) -> list:
    """Get file list with optional caching."""
    import glob
    
    if cache_file and Path(cache_file).exists():
        print(f"Loading cached file list from {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    print(f"Scanning for files (this may take ~30s over NFS)...")
    files = sorted(glob.glob(pattern, recursive=True))
    
    if cache_file:
        print(f"Caching {len(files)} files to {cache_file}")
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(files, f)
    
    return files


def add_ubc_wrf_timesteps(ds):
    """
    Preprocess WRF metgrid files with dummy Time coordinate.
    Extracts date from filename and creates proper time axis.
    Drops first timestep (spin-up).
    """
    # Get filename from dataset encoding
    filepath = ds.encoding.get('source', '')
    
    # Extract year and month: metgrid_YYYY_MM.nc
    match = re.search(r'metgrid_(\d{4})_(\d{2})\.nc$', filepath)
    
    if match:
        year, month = match.groups()
        start_date = f"{year}-{month}-01"
        
        # Drop first timestep (spin-up)
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
            # Use cache to avoid slow glob over NFS
            cache_file = f"/tmp/wrf_files_cache_{hash(path)}.pkl"
            file_list = get_ubc_wrf_file_list(path, cache_file=cache_file)
        else:
            file_list = path
        
        print(f"Opening {len(file_list)} files...")
        ds = xr.open_mfdataset(
            file_list,
            engine=engine,
            parallel=True,
            chunks='auto',
            preprocess=add_ubc_wrf_timesteps,
            combine='nested',
            concat_dim='Times',
            combine_attrs='override',
            data_vars='minimal',      # Add these
            coords='minimal',          # for speed
            compat='override'
        )
        return ds
    else:
        ds = xr.open_dataset(path, engine=engine, chunks=chunks)
        return add_ubc_wrf_timesteps(ds)
    

    