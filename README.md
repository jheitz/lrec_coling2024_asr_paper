# The Influence of Automatic Speech Recognition on Linguistic Features and Automatic Alzheimer’s Disease Detection from Spontaneous Speech

## Overview

This project contains code used to produce our submission to the LREC-Coling 2024 conference.

## Project structure
- `analysis`: Jupyter notebooks for data analysis and plotting of results.
- `cache`: Directory where cached data is written to.
- `conda`: Conda environment files (Python dependencies) 
- `configs`: YAML files representing different configurations of pipeline runs, i.e. different data, processing steps, models, etc.
- `keys`: Directory with keys or other authentication files for external APIs
- `src`: Main directory for source code
  - `src/config`: Config logic and constants for project
  - `src/dataloader`: Dataloader for different datasets and versions
  - `src/evaluation`: Evaluation metrics
  - `src/model`: Implementation of different models
  - `src/preprocessing`: Data preprocessing and transformation steps
  - `src/run`: Helper scripts to run different versions of the pipeline
  - `src/test`: Unit tests
  - `src/util`: Util and helper functions
  - `main.py`: Main pipeline script

## Data
This project depends on the [ADReSS dataset](https://luzs.gitlab.io/adress/). Make sure you have access to this data.

## Setup
### Create conda environment
Create a conda environment from the environment file, specifying all dependencies.
`conda env create -f conda/environment_manual.yml`

If you run into trouble with the dependencies, check the automatically generated `experiment_science_cloud` environment file, which gives precise dependency version used on UZH's ScienceCloud infrastructure (Linux)

### Entry points to running the pipeline
`src/main.py` contains the main Python logic for the pipeline. This reads some command line arguments specifying what and how to run. Usually, you don't call this file directly.

Instead, you should call `src/run/run.py`. This script is a wrapper around `src/main.py`, which does the following
- It runs all unit tests
- It creates a new directory to hold the run's output files (results and logging). The root directory where this is created is defined in `src/config/constants.py`
- It updates the `conda/environment_science_cloud.yaml` file with the latest conda dependency versions, for reproducibilty
- It commits and pushes the current code base to the git experiment branch (defined in `src/config/constants.py`), and stores the commit hash to the results directory, for reproducibilty.
- It calls `src/main.py` to run the pipeline

`src/run/run.py` also takes some command line arguments (most importantly the config file). To make it easier to deal with,
there's `src/run/run.sh`, a bash script defining the commands and command line arguments to use.

### Prepare the code base to contribute
If you want to contribute or reproduce results, you should first complete the following steps

1) Create a new branch for your development
2) Update the constants in `src/config/constants.py`, especially the paths to the data, the experiment branch, and results root path.

### Run the pipline
Edit `src/run/run.sh` to add / uncomment the version of the pipeline you wish to run.

Then run `cd src/run; sh run.sh`.
