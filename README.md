<p align="center">
  <img src="https://user-images.githubusercontent.com/10455520/280422419-5f4c4a78-5811-439d-b861-9d193ffb8902.png" width="250" height="250" /> 
</p>

![example workflow](https://github.com/nannau/ClimatExPrep/actions/workflows/python-package-conda.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/nannau/nc2pt/graph/badge.svg?token=XXRLD3076V)](https://codecov.io/gh/nannau/nc2pt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# Overview

## The Problem
NetCDF4 files, commonly used for storing climate and earth systems data, are not optimized for use with most machine learning applications with heavy io requirements or datasets that are simply too large to hold in GPU/CPU memory. 

## How does nc2pt help?
It performs a preprocessing flow on climate fields and converts them from NetCDF4 (`.nc`) to an intermediate file format Zarr (`.zarr`) which allows for the parallel loading and writing to individual PyTorch Lightning files (`.pt`) that can be loaded directly onto GPUs.

## What are the intended use cases of nc2pt?
  **`nc2pt` prepares climate datasets for machine learning workflows.** It comes pre-configured for downscaling but supports any scenario requiring aligned, ML-ready NetCDF datasets.
  
  ### Primary Use Cases

| Use Case | Description | 
|----------|-------------| 
 **Downscaling Model Training** | Prepare paired low-resolution input and high-resolution target data for training super-resolution models (GANs, CNNs, diffusion models). Includes feature scaling (normalization/standardization) and train/val/test splitting. |
 | **Evaluation Dataset Preparation** | Map multiple high-resolution datasets (model predictions, reference data, ensemble members) onto a common grid with consistent spatial extent, resolution, and temporal coverage for fair comparison. | 
| **Custom ML Preprocessing** | Build bespoke preprocessing pipelines with configurable alignment steps, transformations, and normalization schemes for research-specific ML applications. | 

### What nc2pt Provides

**Core Preprocessing:** Spatial alignment, temporal synchronization, feature scaling, metadata standardization, train/val/test splitting

**Format & Performance:** NetCDF → Zarr → PyTorch conversion, parallelized I/O for O(terabyte) datasets, optional batching for reduced I/O overhead

**Customization:** Configurable processing pipelines, custom transformations (unit conversion, log transforms, user functions)


## Understanding `ClimateModel` Objects

**`nc2pt` organizes datasets using `ClimateModel` configurations.** Each `ClimateModel` defines a preprocessing workflow for a category of data (e.g., low-resolution inputs, high-resolution targets, model predictions).

**Why separate `ClimateModels`?** Different data sources need different preprocessing. Low-resolution data needs regridding and coarsening to match high-resolution targets. Static fields need no temporal processing. Model outputs may need different normalization schemes.

**Default setup**: `nc2pt` comes with pre-configured `lr` (low-resolution input), `hr` (high-resolution target), `hr_invariant`(invariant high-resolution input), and `lr_emulation` (low-resolution input for inference) `ClimateModels` for downscaling workflows.

## What preprocessing steps does nc2pt do? 🤔

**Example: The `lr` (low-resolution) `ClimateModel` workflow**

1. **Configure metadata** - Standardize variable names and attributes across datasets
2. **Temporal crop** - Slice data to specified date range
3. **Regrid** - Interpolate onto target grid coordinates
4. **Spatial crop** - Crop to specified spatial extent
5. **Coarsen** - Reduce resolution by scale factor to return to desired (often native) coarseness
6. **User-defined transforms** - Apply unit conversions or custom functions
7. **Data split** - Partition into train/test/validation by year
8. **Feature scaling** - Normalize/standardize using training statistics (statistics saved to metadata file for post-processing)
9. **Write to Zarr** - Save intermediate format with preserved metadata
10. **Serialize to PyTorch** - Convert to `.pt` files via `nc2pt/tools/zarr_to_torch.py`
11. **Optional batching** - Combine files via `nc2pt/tools/single_file_to_batches.py`

**The `hr` (high-resolution) `ClimateModel` differs:** Skips steps 3, 5 (regrid/coarsen) since it defines the target grid that `lr` aligns to.

## Customizable Pipelines 🚦

Each `ClimateModel`'s preprocessing steps are configurable via `alignment_pipeline`:

```yaml
# conf/climate_models/lr.yaml
alignment_pipeline:
  - temporal_crop
  - regrid
  - spatial_crop
  - coarsen
  - user_defined_transforms
  - data_split
```

**Customize by removing or reordering steps** to match your workflow:

```yaml
#Example: Skip coarsening and data split for model output already at target resolution
alignment_pipeline:
  - temporal_crop
  - regrid
  - spatial_crop
 ```


# Using nc2pt

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
    Place it at `conf/climate_models/<model_name>.yaml`. Example:
    
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
    In `conf/climate_models/hr.yaml`:
    
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
