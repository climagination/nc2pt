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
5. optionally splits into train/test/validation based on years defined in the config
6. standardizes/normalizes the datasets based on statistics computed from the training data (or full data, if no split). These statistics are stored in the metadata.
7. writes to `.zarr`
8. `nc2pt/tools/zarr_to_torch.py` - writes to PyTorch files
9. `nc2pt/tools/single_file_to_batches.py` - batches the single PyTorch files

## Customizable Pipelines 🚦

Each model can define its own custom preprocessing steps by listing them in order via the `alignment_pipeline` field of the model YAML. Steps include:

-   `temporal_crop`
    
-   `regrid`
    
-   `spatial_crop`
    
-   `coarsen`

-   `user_defined_transforms`
    
-   `data_split`
    

By default, all 6 are applied. You can exclude or reorder them by editing the YAML, e.g.:

```
alignment_pipeline:
  - temporal_crop
  - regrid
  - spatial_crop
```

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

`nc2pt` uses [Hydra](https://hydra.cc/) for flexible configuration. The main configuration file is `conf/config.yaml`, which defines:

-   The list of climate models to include (`climate_models`)
    
-   Global dimensions, coordinates, subsetting, and chunking
    
-   Output path and compute options
    

Each model and variable is defined in separate YAML files under `conf/climate_models/`, making the pipeline modular and easily extensible.

----------

### ➕ Adding a New Climate Model

To add a new model:

1.  **Create a model file**  
    Place it at `conf/climate_models/<name>/model.yaml`. Example:
    
    ```yaml
    _target_:  nc2pt.climatedata.ClimateModel
    name:  my_model
    info:  "My custom climate model"
	alignment_pipeline:
      - temporal_crop
      - regrid
      - spatial_crop
      - coarsen
	  - user_defined_transforms
      - data_split
    climate_variables:
	    -  ${internal.my_model_pr}
	    -  ${internal.my_model_tas}
    ```
    
2.  **Register it in `injections.yaml`**
    
	``` yaml
	default:
		-  climate_models/my_model@internal._my_model  
		-  climate_models/my_model/pr@internal._my_model_pr
		-  climate_models/my_model/tas@internal._my_model_tas 
	```
	
3.  **Expose the aliases in `injections.yaml`**
    
	```yaml
	internal:
		my_model:  ${internal._my_model}
		my_model_pr:  ${internal._my_model_pr}
		my_model_tas:  ${internal._my_model_tas} 
	 ``` 
4.  **Enable it in `config.yaml`**
    
	   ``` yaml
	  climate_models:
		  -  ${internal.my_model}
	  ```
    

----------

### ➕ Adding a New Climate Variable

To add a new variable to an existing model (e.g., `hr`):

1.  **Create a variable file**  
    Place it at `conf/climate_models/hr/zg.yaml`:
    
	   ``` yaml
	    _target_:  nc2pt.climatedata.ClimateVariable
	    name:  "zg"
	    alternative_names: ["zg"]
	    path:  ${internal.paths.hr.zg}
	    apply_standardize:  true
	    apply_normalize:  true
	    invariant:  false
		transform: ["x * 69 + 420"]
	 ```
    
2.  **Register and alias it in `injections.yaml`**
    
	``` yaml
	defaults:
		- climate_models/hr/zg@internal._hr_zg
	internal:
		hr_zg:  ${internal._hr_zg}
	```

3.  **Add it to the model’s variable list**  
    In `conf/climate_models/hr/model.yaml`:
    
	   ``` yaml
	    climate_variables:
		    -  ${internal.hr_zg}  # other variables...
	  ```
    

That’s it — your new model or variable will now be included in the pipeline when `preprocess.py` is run.

### 🚀 Running
1. Explore data and ensure compatibility
2. **Set up your configuration**:
-   Edit `conf/config.yaml` to include the models you want to use under `climate_models:`   
-   For each model, go to its `model.yaml` file and uncomment (or add) the variables you want included
3. Run the `nc2pt/preprocess.py` script which will run through your preprocessing steps. This creates the zarr files
4. Run the `nc2pt/tools/zarr_to_torch.py` script which serializes each time step in the `.zarr` file to an individual PyTorch `.pt` file.
5. Optional: run the `nc2pt/tools/single_files_to_batches.py` which combines individual files from the previous step into random batches. This setup allows for less io in your machine learning pipeline.

### Testing

Testing is done with pytest. The easiest way to perform tests is to install pytest and use the command: `pytest --cov-report term-missing --cov=nc2pt .`

It will generate a coverage report and automatically use files prepended with `test_*.py` in `nc2pt/tests`


### 📝 Notes

- **Chunking Sensitivity**:  
  The preprocessing pipeline is sensitive to how datasets are chunked in memory. If you encounter memory errors or Dask worker crashes, reviewing and adjusting the chunk sizes is a good first step. See [closed issue #18](https://github.com/nannau/nc2pt/issues/18) for details and suggestions.

- **Interpolation Method**:  
  The current interpolation method uses xarray’s native 2D interpolation, which does not account for Earth curvature. This repository previously used an `xESMF`-backed interpolation scheme that performed regridding on spherical geometry. However, within the scope of this work, it was found that the difference in performance was negligible, so the dependency on `xESMF` was removed. See [closed issue #15](https://github.com/nannau/nc2pt/issues/15) for more context.
