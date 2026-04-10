<p align="center">
  <img src="https://user-images.githubusercontent.com/10455520/280422419-5f4c4a78-5811-439d-b861-9d193ffb8902.png" width="250" height="250" /> 
</p>

![example workflow](https://github.com/nannau/ClimatExPrep/actions/workflows/python-package-conda.yml/badge.svg?event=push)
[![codecov](https://codecov.io/gh/nannau/nc2pt/graph/badge.svg?token=XXRLD3076V)](https://codecov.io/gh/nannau/nc2pt)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)


![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# nc2pt: NetCDF to PyTorch for Climate Data

**Convert NetCDF climate data to PyTorch-ready formats for machine learning**

---

## Table of Contents
- [Quick Start](#quick-start)
- [What is nc2pt?](#what-is-nc2pt)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
  - [Your First Preprocessing Run](#your-first-preprocessing-run)
  - [Verifying Output](#verifying-output)
- [Core Concepts](#core-concepts)
  - [ClimateModel Objects](#climatemodel-objects)
  - [ClimateVariable Objects](#climatevariable-objects)
  - [Processing Pipeline](#processing-pipeline)
- [Configuration Guide](#configuration-guide)
  - [Essential Configuration Files](#essential-configuration-files)
  - [Understanding the Config Structure](#understanding-the-config-structure)
- [Common Workflows](#common-workflows)
  - [Downscaling (Default)](#downscaling-default)
  - [Model Evaluation](#model-evaluation)
  - [Custom ML Preprocessing](#custom-ml-preprocessing)
- [Customization](#customization)
  - [Adding Models](#adding-a-new-climatemodel)
  - [Adding Variables](#adding-a-new-climatevariable)
  - [Customizing Processing Pipelines](#customizable-pipelines)
- [Advanced Topics](#advanced-topics)
  - [Memory and Chunking](#memory-and-chunking)
  - [Scaling Statistics and Inference](#scaling-statistics-and-inference)
- [Technical Notes](#technical-notes)


---

## Quick Start

Get your first NetCDF → PyTorch conversion running in 5 minutes:

### 1. Install
```bash
git clone https://github.com/climagination/nc2pt.git
cd nc2pt
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

### 2. Point to Your Data
Edit `conf/paths.yaml` with paths to your NetCDF files:

```yaml
paths:
  hr_ref: /path/to/high_res_reference.nc  # Single-timestep HR grid
  
  hr:
    tas: /path/to/high_res_temperature*.nc
  
  lr:
    tas: /path/to/low_res_temperature*.nc
```

### 3. Configure Your Domain

Edit `conf/select.yaml` to set your spatial/temporal extent and test/train/validation split:

```yaml
select:
  time:
    range:
      start: "2015-01-01"
      end: "2016-01-01"

	#Select years to reserve for testing and validation
	#Remaining years are used for training
	test_years: [2014]
	validation_years: [2015]

  spatial:
    scale_factor: 8  # LR will be 8x coarser than HR
    x:
      first_index: 0
      last_index: 128
    y:
      first_index: 0
      last_index: 128
  ```

### 4. Run Preprocessing

``` bash
python nc2pt/preprocess.py 
```

This creates `.zarr` files in your output directory (default: `output/`).

### 5. Convert to PyTorch
```bash
python nc2pt/tools/zarr_to_torch.py
```

**Done!** You now have individual `.pt` files for each timestep.

### 📦 What You Just Created

```bash
output/
├── hr/
│   └── tas/
│       ├── train/
│       │   ├── 20150101_00.pt
│       │   ├── 20150101_01.pt
│       │   └── ...
│       └── test/
│           └── ...
└── lr/
    └── tas/
        └── ...` 
```
Each `.pt` file contains a single timestep ready to load directly onto your GPU:

``` python
import torch
data = torch.load('output/hr/tas/train/20150101_00.pt')
print(data.shape)  # (1, height, width)
```

### 🚨 Troubleshooting

**"FileNotFoundError: No such file"**

-   Check your paths in  `conf/paths.yaml`  are correct
-   Verify wildcards (`*.nc`) match your filenames

**"Memory Error" or "Worker crashed"**

-   Reduce  `n_workers`  in  `conf/compute.yaml`  (try  `n_workers: 2`)
-   See  [Memory and Chunking](#memory-and-chunking)  for chunking guidance

**"Variable 'tas' not found"**
-   Check your NetCDF variable names with  `ncdump -h yourfile.nc`
-   Add them to  `alternative_names`  in  `conf/climate_models/hr/tas.yaml`

### 🎯 Next Steps
-   **Add more variables**: See  [Adding Climate Variables](#adding-a-new-climatevariable)
-   **Customize preprocessing**: Learn about  [ClimateModel pipelines](#customizable-pipelines)
-   **Train a model**: Check out our  [GAN for downscaling](https://github.com/climagination/ClimatExML)

## What is nc2pt?
nc2pt is a preprocessing tool that converts NetCDF climate datasets into PyTorch-ready formats optimized for machine learning workflows.

## The Problem
NetCDF4 files, commonly used for climate and earth systems data, are not optimized for:
- **ML training loops** with heavy I/O requirements
- **Datasets that exceed GPU/CPU memory** (terabyte-scale)
- **Fast random access** during batched training 

## The Solution
nc2pt provides:

| Feature | Benefit |
|---------|---------|
| **Format Conversion** | NetCDF → Zarr → PyTorch for GPU-ready data |
| **Spatial Alignment** | Automatic regridding and domain matching |
| **Feature Scaling** | Built-in normalization/standardization |
| **Data Splitting** | Train/validation/test splits by year |
| **Parallel I/O** | Process terabyte-scale datasets efficiently |


### When to Use nc2pt
✅ **Use nc2pt when:**
- Training downscaling models (GANs, CNNs, diffusion models)
- Preparing evaluation datasets with consistent grids
- Building custom ML pipelines with climate data
- Working with datasets too large for memory

## Installation

### Requirements
- Python >= 3.8
- Recommended: virtual environment (e.g. `venv` or `virtualenv`)

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

## Basic Usage

### Your First Preprocessing Run
This section walks through preprocessing a single variable (temperature) with the default downscaling configuration.

#### Prerequisites
You need:
- NetCDF files with temperature data at two resolutions (high-res and low-res)
- ~1-2x the size of your input data available as disk space

#### Step-by-Step
**1. Configure file paths** (`conf/paths.yaml`):
```yaml
paths:
  hr_ref: /data/reference/hr_grid_single_timestep.nc
  
  hr:
    tas: /data/high_res/temperature_*.nc
  
  lr:
    tas: /data/low_res/temperature_*.nc
```
**2. Set your domain** (`conf/select.yaml`):
```yaml
select:
  time:
    range:
      start: "2015-01-01"
      end: "2017-01-01"
    test_years: [2016]
    validation_years: [2015]
  
  spatial:
    scale_factor: 8  # LR will be 8x coarser than HR
    x:
      first_index: 0
      last_index: 256
    y:
      first_index: 0
      last_index: 256
```
**3. Enable only temperature** (`conf/climate_models/hr.yaml`):
``` yaml
climate_variables:
  - ${internal.hr_tas}
  # - ${internal.hr_pr}   # Comment out other variables
  # - ${internal.hr_uas}
```
Do the same in conf/climate_models/lr.yaml.

**4. Run preprocessing**:
```bash
python nc2pt/preprocess.py
```

**5. Convert to PyTorch**:
``` bash
python nc2pt/tools/zarr_to_torch.py
```

### Verifying Output

After successful processing, check your output:

``` bash
# Check zarr files exist
ls -lh output/*.zarr

# Check PyTorch files
find output -name "*.pt" | head -5

# Count files per split
ls output/hr/tas/train/*.pt | wc -l
ls output/hr/tas/val/*.pt | wc -l
ls output/hr/tas/test/*.pt | wc -l
```

## Core Concepts

Now that you've run your first preprocessing, let's understand the key abstractions.

### ClimateModel Objects
A **ClimateModel** defines a preprocessing workflow for a category of data (e.g., low-resolution inputs, high-resolution targets).

#### Why Separate ClimateModels?

Different data sources need different preprocessing:

| Model | Purpose | Key Differences|
|---------|---------|---------|
| `hr` | High-resolution target data | Defines the target grid, no regridding needed|
| `lr` | Low-resolution input data | Regridded to HR grid, then coarsened back to native resolution|
| `hr_invariant` | Time-invariant fields (topography) | No temporal processing, used as static features|
| `lr_emulation` | Inference-time data | Uses pre-computed scaling statistics, no train/test split|

#### ClimateModel Configuration

Each ClimateModel is defined in `conf/climate_models/<model_name>.yaml`:

```yaml
_target_: nc2pt.climatedata.ClimateModel
name: lr
info: "Low-resolution input data for downscaling"

alignment_pipeline:
  - temporal_crop
  - regrid
  - spatial_crop
  - coarsen
  - user_defined_transforms
  - data_split

climate_variables:
  - ${internal.lr_tas}
  - ${internal.lr_pr}
```
The `alignment_pipeline` defines which processing steps to apply and in what order.

### ClimateVariable Objects
A **ClimateVariable** represents a single physical field (temperature, precipitation, wind) with its processing metadata. Each ClimateVariable is defined in `conf/climate_models/<model_name>/<variable_name>.yaml`.

#### Key Attributes
```yaml
_target_: nc2pt.climatedata.ClimateVariable
name: "tas"                              # Standard variable name
alternative_names: ["T2", "t2m", "temp"] # Names it might have in your files
path: ${internal.paths.hr.tas}           # File path
is_west_negative: false                  # Longitude convention
invariant: false                         # Time-varying or static?
apply_standardize: true                  # Zero mean, unit variance
apply_normalize: false                   # Min-max scaling to [0,1]
coarsening_method: "mean"                # How to aggregate when coarsening
transform: []                            # Custom preprocessing (see below)
```

#### Example: Unit Conversion
Precipitation often needs conversion from kg/m²/s to mm/hr:
``` yaml
# conf/climate_models/hr/pr.yaml
name: "pr"
transform: ["x * 3600"]  # kg/m²/s → mm/hr
apply_normalize: true    # [0,1] scaling works better for precipitation
coarsening_method: "sum" # Preserve total precipitation when coarsening
```

### Processing Pipeline
The preprocessing pipeline consists of configurable steps that transform your data:

#### Available Pipeline Steps
| Step | What it does | When it's applied|
|---------|---------|---------|
| `configure_metadata` | Standardize variable/dimension names | Always (automatic)|
| `temporal_crop` | Slice to date range | Based on `select.yaml`|
| `regrid` | Interpolate to target grid coordinates | Only for `lr` model|
| `spatial_crop` | Extract spatial domain subset | Based on `select.yaml`|
| `coarsen` | Reduce resolution by scale factor | Only for `lr` model|
| `user_defined_transforms` | Apply unit conversions, log transforms | If specified in variable config|
| `data_split` | Partition into train/val/test by year | Based on `select.yaml`|
| `feature scaling` | Normalize or standardize | If enabled in variable config|
| `write_to_zarr` | Save intermediate format | Always (automatic)|

## Configuration Guide

### Essential Configuration Files
These are the files you'll edit most often:
#### Priority 1: Always Edit
| File | Purpose | What to Change | 
|------|---------|----------------|
| `config.yaml` | Enable/disable models | Uncomment models under `climate_models:` |
| `paths.yaml` | Point to your NetCDF files | All file paths |
| `select.yaml` | Define domain and splits | Date range, spatial extent, test/val years |

#### Priority 2: Often Edit
| File | Purpose | What to Change | 
|------|---------|----------------|
| `climate_models/hr.yaml` | Configure HR model | Enable/disable variables, modify pipeline |
| `climate_models/lr.yaml` | Configure LR model | Enable/disable variables, modify pipeline |
| `climate_models/hr/<var>.yaml` | Variable-specific settings | Scaling method, transforms, units |

#### Priority 3: Rarely Edit
| File | Purpose | When to Edit |
|------|---------|-------------|
| `compute.yaml` | Dask parallelization | Memory errors, performance tuning |
| `dims.yaml` | Dimension name mappings | Non-standard NetCDF dimension names |
| `coords.yaml` | Coordinate name mappings | Non-standard coordinate variable names |

#### Priority 4: Advanced Only
| File | Purpose | When to Edit |
|------|---------|-------------|
| `injections.yaml` | Hydra dependency injection | Adding new models or variables |
| `loader.yaml` | Training batch loader | ML training setup |

### Understanding the Config Structure
The configuration uses **Hydra's composition pattern**:

``` yaml
# config.yaml references models
climate_models:
  - ${internal.hr}    # References hr.yaml
  - ${internal.lr}    # References lr.yaml

# hr.yaml references variables
climate_variables:
  - ${internal.hr_tas}   # References hr/tas.yaml
  - ${internal.hr_pr}    # References hr/pr.yaml

# hr/tas.yaml references paths
path: ${internal.paths.hr.tas}   # References paths.yaml
```

**Key insight:** The `${internal.*}` syntax creates references that are resolved at runtime by Hydra.

## Common Workflows

### Downscaling (Default)

This is the pre-configured workflow. You've already run this in [Your First Preprocessing Run](#your-first-preprocessing-run).

**Use case:** Training super-resolution models to downscale coarse climate model output.

**What's included:**

-   `lr`: Low-resolution inputs (regridded and coarsened)
-   `hr`: High-resolution targets
-   `hr_invariant`: Static fields like topography (optional)

**Configuration:** Already set up in `conf/climate_models/`.

### Model Evaluation
**Use case:** Comparing multiple model outputs on a common grid for fair evaluation.

**Scenario:** You have multiple datasets (observations, model predictions, ensemble members) that may have different resolutions, projections, or spatial extents. You need them all on a common grid with identical spatial and temporal coverage for fair comparison.

**Setup:** Use the `lr` ClimateModel (which supports regridding to a reference grid).

#### Configuration

**1. Set your reference grid** (`conf/paths.yaml`):

```yaml
paths:
  hr_ref: /data/reference/grid.nc  # Defines target grid for all datasets
  ```

**2. Disable scaling and data splitting**

For each variable (e.g., `conf/climate_models/lr/tas.yaml`, `conf/climate_models/lr/pr.yaml`):

```yaml
apply_standardize: false  # Keep raw values for comparison
apply_normalize: false` 
```


**3. Remove coarsening and data split** (`conf/climate_models/lr.yaml`):

```yaml
alignment_pipeline:
  - temporal_crop
  - regrid        # Aligns all datasets to hr_ref grid
  - spatial_crop
  - user_defined_transforms
  # - coarsen     # REMOVE - output at target resolution
  # - split_data  # REMOVE - no train/val/test split needed
```
**4. Enable all variables you want to compare** (`conf/climate_models/lr.yaml`):

``` yaml
climate_variables:
  - ${internal.lr_tas}
  - ${internal.lr_pr}
  - ${internal.lr_uas}
  # Add all variables you need to evaluate
  ```
**5. Process each dataset:**
``` bash
# Reference dataset (observations)
# Edit conf/paths.yaml:
#   lr.tas: /data/observations/temperature_*.nc
#   lr.pr: /data/observations/precipitation_*.nc
#   lr.uas: /data/observations/wind_u_*.nc
python nc2pt/preprocess.py
mv output/lr output/reference

# Model A predictions
# Edit conf/paths.yaml:
#   lr.tas: /data/model_a/temperature_*.nc
#   lr.pr: /data/model_a/precipitation_*.nc
#   lr.uas: /data/model_a/wind_u_*.nc
python nc2pt/preprocess.py
mv output/lr output/model_a

# Model B predictions
# Edit conf/paths.yaml with Model B paths
python nc2pt/preprocess.py
mv output/lr output/model_b

# Repeat for additional models...
```

#### Result
All datasets are now:
-   ✅ On the same grid (resolution and projection from  `hr_ref`)
-   ✅ Same spatial extent (from  `select.yaml`  spatial crop)
-   ✅ Same temporal coverage (from  `select.yaml`  time range)
-   ✅ Same variables with consistent units
-   ✅ Ready for direct comparison

#### Final Directory Structure

```bash
output/
├── reference/
│   ├── tas/
│   ├── pr/
│   └── uas/
├── model_a/
│   ├── tas/
│   ├── pr/
│   └── uas/
└── model_b/
    ├── tas/
    ├── pr/
    └── uas/
```

### Custom ML Preprocessing

**Use case:** You have a specific ML task that doesn't fit the downscaling template.

**Example:** Bias correction with multiple predictors.

**Approach:**

1.  Create custom ClimateModels for each data source
2.  Define custom  `alignment_pipeline`  for each
3.  Use  `user_defined_transforms`  for feature engineering


## Customization

### Adding a New ClimateModel

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

### Adding a New ClimateVariable

First check whether the variable already exists in your ClimateModel! (Check `conf/climate_models/<model_name>.yaml`, your variable might just be commented out).

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

### Customizable Pipelines

Each ClimateModel's `alignment_pipeline` can be customized by removing or reordering steps.

#### Example 1: Skip Coarsening

If your LR data is already at the desired resolution:

```yaml
# conf/climate_models/lr.yaml
alignment_pipeline:
  - temporal_crop
  - regrid
  - spatial_crop
  # - coarsen  ← REMOVED
  - user_defined_transforms
  - data_split
```

#### Example 2: No Train/Test Split

For inference or when doing custom cross-validation:

``` yaml
alignment_pipeline:
  - temporal_crop
  - regrid
  - spatial_crop
  - coarsen
  - user_defined_transforms
  # - data_split  ← REMOVED` 
```

#### Example 3: Minimal Pipeline

For data that's already perfectly aligned:

```yaml
alignment_pipeline:
  - temporal_crop
  - user_defined_transforms
  - data_split
```

## Advanced Topics

### Memory and Chunking
nc2pt uses Dask for parallel processing, which requires careful chunking for large datasets.
#### Common Memory Issues
**Symptom:** `MemoryError` or `KilledWorker` during preprocessing.
**Solutions:**

1.  **Reduce workers**  (`conf/compute.yaml`):
```yaml
compute:
  n_workers: 2  # Default is 4
```

2.  **Adjust chunking**  (`conf/compute.yaml`):

```yaml
compute:
  chunk_size:
    time: 10    # Process 10 timesteps at a time
    x: 128      # Spatial chunks
   y: 128
```
**See also:**  [Issue #18](https://github.com/climagination/nc2pt/issues/18) for detailed discussion.

### Scaling Statistics and Inference

When using standardization/normalization, statistics are computed from the training set and saved:

```bash
output/
└── feature_scaling_metadata/
    └── hr_tas_feature_scaling_metadata.json
```

**Content:**

```json
"method": "minmax",
"min": -22.145673751831055,
"max": 30.749866485595703,
```
#### Using at Inference Time
For the `lr_emulation` model (inference without train/test split):
```yaml
# conf/climate_models/lr_emulation/tas.yaml
metadata_path: /path/to/hr_tas_feature_scaling_metadata.json
```

This applies the training set statistics to your inference data, ensuring consistent preprocessing.

## Technical Notes

### Interpolation Method

The current implementation uses xarray's native 2D interpolation, which does not account for Earth curvature. Previous versions used xESMF for spherical regridding, but performance differences were negligible for typical regional domains. See [Issue #15](https://github.com/climagination/nc2pt/issues/15) for context.

### Chunking Sensitivity

Preprocessing performance is sensitive to chunk sizes. The default configuration works well for typical datasets, but large domains or high temporal resolution may require tuning. See [Issue #18](https://github.com/climagination/nc2pt/issues/18) for guidelines.
