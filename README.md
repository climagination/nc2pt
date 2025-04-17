<p align="center">
  <img src="https://user-images.githubusercontent.com/10455520/280422419-5f4c4a78-5811-439d-b861-9d193ffb8902.png" width="250" height="250" /> 
</p>

![example workflow](https://github.com/nannau/ClimatExPrep/actions/workflows/python-package-conda.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/nannau/nc2pt/graph/badge.svg?token=XXRLD3076V)](https://codecov.io/gh/nannau/nc2pt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## The Problem
NetCDF4 files, commonly used for storing climate and earth systems data, are not optimized for use with most machine learning applications with heavy io requirements or datasets that are simply too large to hold in GPU/CPU memory. 

## How does nc2pt help?
It performs a preprocessing flow on climate fields and converts them from NetCDF4 (`.nc`) to an intermediate file format Zarr (`.zarr`) which allows for the parallel loading and writing to individual PyTorch Lightning files (`.pt`) that can be loaded directly onto GPUs.

## What intended use cases of nc2pt?
* standardizing and making metadata uniform between datasets

* aligns different grids perfectly by re-projecting them onto one another -- nc2pt projects the low-resolution (lr) regular grids onto high-resolution curvilinear grids (hr). This step can be configured to suit specific datasets.

* selects individual years as test years or training years

* organizes code into input (lr) or output (hr) fields

* Designed for O(terabyte) datasets

## What preprocessing steps does nc2pt do? 🤔

High-level workflow
![image](https://private-user-images.githubusercontent.com/10455520/313314202-e13396ce-2224-4298-8fa2-472631efe4df.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3MTQ2Nzc3NDksIm5iZiI6MTcxNDY3NzQ0OSwicGF0aCI6Ii8xMDQ1NTUyMC8zMTMzMTQyMDItZTEzMzk2Y2UtMjIyNC00Mjk4LThmYTItNDcyNjMxZWZlNGRmLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDA1MDIlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQwNTAyVDE5MTcyOVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTA0MzAxN2JiZGMyZjYwNzcyYTEwNjgxYjRhNmNiYTEyNDMzMTlmZDRiZWVkYTAzZjcyNWU0YTFhNDI2NGUzNzAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0JmFjdG9yX2lkPTAma2V5X2lkPTAmcmVwb19pZD0wIn0.xcTKPMiumJhWMo9TzvBG499QOPQPvaI7XekUifjwXLs)

1. configures metadata between the datasets as defined in the config
2. slices data to a pre-determined range of dates
3. aligns the grids via interpolation, crops them to be the same size, and coarsens the low-resolution fields by the configured scale factor
4. applies user defined transforms like unit conversions or log transformations
5. splits into a train and test dataset and standardizes both datasets based on the mean and standard deviation of all grids from the training data only (also writes this information into the zarr metadata for inference)
6. writes to `.zarr`
7. `nc2pt/tools/zarr_to_torch.py` - writes to PyTorch files
8. `nc2pt/tools/single_file_to_batches.py` - batches the single PyTorch files

## What are the downsides of using PyTorch files for climate data?
The most obvious downside is that you lose the metadata associated with a netCDF dataset. The intermediate Zarr format produced by nc2pt allows for parallelized io and perserves the metadata. This is useful for inference. 


## Requirements

- Python >= 3.8
- Recommended: virtual environment (e.g. `venv` or `virtualenv`)

## 💽 Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/climagination/nc2pt.git
   cd nc2pt
   ```

2. (Optional but recommended) Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in editable mode:

   ```bash
   pip install -e nc2pt/
   ```

That’s it!

### 📋 Configuration
nc2pt uses [hydra](https://hydra.cc/) for configuring and by instantiating structured classes in `nc2pt/climatedata.py`. This simeultaneously defines the workflow as well as the data. Please see `nc2pt/conf/config.yml` for an example configuration, or below:

```yaml
_target_: nc2pt.climatedata.ClimateData # Iniatlizes ClimateData dataclass object
output_path: /home/sbeairsto/data-sbeairsto/test_data
climate_models:
  - _target_: nc2pt.climatedata.ClimateModel
    name: hr
    info: "High Resolution USask WRF, Western Canada"
    climate_variables: # Provides a list of ClimateVariable dataclass objects to initialize

       - _target_: nc2pt.climatedata.ClimateVariable
         name: "uas"
         alternative_names: ["U10", "u10", "uas"]
         path: /net/venus/kenes/downloaded-data/acannon/USask-WRF-WCA/pgw-wrf-wca/U10/*.nc
         is_west_negative: true
         apply_standardize: false
         apply_normalize: true
         invariant: false
         transform: []
  

  - _target_: nc2pt.climatedata.ClimateModel
    info: "Low resolution ERA5, Western Canada"
    name: lr
    hr_ref: # Reference field to interpolate to. Will need to provide new file if not using USask WRF
      _target_: nc2pt.climatedata.ClimateVariable
      name: "hr_ref"
      alternative_names: ["T2"]
      path: /home/sbeairsto/projects/nc2pt/nc2pt/data/hr_ref.nc
      is_west_negative: true
      invariant: true

    climate_variables:
       - _target_: nc2pt.climatedata.ClimateVariable
         name: "uas"
         alternative_names: ["U10", "u10", "uas"]
         path: /net/venus/kenes/downloaded-data/acannon/ERA5_NCAR-RDA_North_America/1hr/uas_1hr_ERA5_an_RDA-025_1979010100-2018123123.nc
         is_west_negative: false
         apply_standardize: false
         apply_normalize: true
         invariant: false
         transform: []


dims: # Defines the dimensions of the dataset and lists them to be initialized as ClimateDimension objects
  - _target_: nc2pt.climatedata.ClimateDimension
    name: time
    alternative_names: ["forecast_initial_time", "Time", "Times", "times"]
    chunksize: 100
  - _target_: nc2pt.climatedata.ClimateDimension
    name: rlat
    alternative_names: ["rotated_latitude"]
    hr_only: true
    chunksize: -1
  - _target_: nc2pt.climatedata.ClimateDimension
    name: rlon
    alternative_names: ["rotated_longitude"]
    hr_only: true
    chunksize: -1

coords:
  - _target_: nc2pt.climatedata.ClimateDimension
    name: lat
    alternative_names: ["latitude", "Lat", "Latitude"]
    chunksize: -1
  - _target_: nc2pt.climatedata.ClimateDimension
    name: lon
    alternative_names: ["longitude", "Long", "Lon", "Longitude"]
    chunksize: -1

select:
  # Time indexing for subsets
  time:
    # Crop to the dataset with the shortest run
    # this defines the full dataset from which to subset
    range:
      start: "20001001T06:00:00"
      end: "20150928T12:00:00"

    # use this to select which years to reserve for testing
    # and for validation
    # the remaining years in full will be used for training
    test_years: [2000, 2009, 2014]
    validation_years: [2015]

  # sets the scale factor and index slices of the rotated coordinates
  spatial:
    scale_factor: 8
    x:
      first_index: 110
      last_index: 622
    y:
      first_index: 20
      last_index: 532



# dask client parameters
compute:
  # xarray netcdf engine
  engine: h5netcdf
  dask_dashboard_address: 8787
  chunks:
    time: 100
    rlat: -1
    rlon: -1

# for tools scripts
loader:
  batch_size: 4
  randomize: true
  seed: 0
```

### 🚀 Running
1. Explore data and ensure compatibility
2. Configure `nc2pt/conf/config.yaml`
3. Run the `nc2pt/preprocess.py` script which will run through your preprocessing steps. This creates the zarr files
4. Run the `nc2pt/tools/zarr_to_torch.py` script which serializes each time step in the `.zarr` file to an individual PyTorch `.pt` file.
5. Optional: run the `nc2pt/tools/single_files_to_batches.py` which combines individual files from the previous step into random batches. This setup allows for less io in your machine learning pipeline.

### Testing

Testing is done with pytest. The easiest way to perform tests is to install pytest and use the command: `pytest --cov-report term-missing --cov=nc2pt .`

It will generate a coverage report and automatically use files prepended with `test_*.py` in `nc2pt/tests`
