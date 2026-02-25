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


def validate_file(filepath, engine='h5netcdf'):
    """Check if a file is readable."""
    try:
        with xr.open_dataset(filepath, engine=engine) as ds:
            # Try to access dimensions (forces reading header)
            _ = ds.dims
            # Try to access a variable's shape if any exist
            if len(ds.data_vars) > 0:
                var_name = list(ds.data_vars.keys())[0]
                _ = ds[var_name].shape
        return True
    except Exception as e:
        print(f"  ⚠️  Skipping: {filepath}")
        print(f"      Error: {str(e)[:80]}")
        return False


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
        if 'Times' in ds.dims and ds.sizes['Times'] <= 2:
            ds = ds.isel(Times=0, drop=True)
    
    return ds


def load_ubc_wrf(path: str, engine: str = "h5netcdf", chunks: dict = None) -> xr.Dataset:
    """Load WRF metgrid files with proper time coordinate handling."""
    
    # Use provided chunks or default to auto
    if chunks is None:
        chunks = 'auto'
    
    if "*" in path or isinstance(path, list):
        if isinstance(path, str):
            file_list = get_ubc_wrf_file_list(path)
        else:
            file_list = path
        
        # Filter for only COMPRESSED_SUBSETTED files
        # file_list = [f for f in file_list if 'COMPRESSED_SUBSETTED_d03' in f]
        # print(f"Filtered to {len(file_list)} COMPRESSED_SUBSETTED files")
        
        # Validate files
        print("Validating files...")
        valid_files = [f for f in file_list if validate_file(f, engine)]
        
        if len(valid_files) < len(file_list):
            print(f"⚠️  Skipped {len(file_list) - len(valid_files)} corrupted files")
        
        print(f"✅ Processing {len(valid_files)} valid files")
        print("Note: Dropping last timestep of each month (corresponds to M+1 00:00:00)")
        
        ds = xr.open_mfdataset(
            paths=valid_files,
            engine=engine,
            parallel=True,
            chunks='auto',
            preprocess=add_ubc_wrf_timesteps,
            combine='nested',
            concat_dim='Times',
            combine_attrs='override',
            data_vars='minimal',
            coords='minimal',
            compat='override',
        )
        return ds
    else:
        ds = xr.open_dataset(path, engine=engine, chunks=chunks)
        return add_ubc_wrf_timesteps(ds)