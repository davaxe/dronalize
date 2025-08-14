
# Preprocessing

The **Dronalize** toolbox provides dataset-specific preprocessing scripts to convert raw trajectory data into a unified, model-ready format, ensuring consistency across datasets.
This page outlines a brief overview of the preprocessing process, including configuration management and usage instructions.
For detailed usage instructions, folder structure expectations, and specific commands for each supported dataset, please refer to [**Datasets**](docs/datasets.md).

All specific preprocessing commands are also available in [`preprocess.sh`](../preprocess.sh) located in the root directory of the repository.


## Overview
All preprocessing logic is located in the [`preprocessing`](preprocessing) module, with dataset-specific configurations defined in [`preprocessing/config`](preprocessing/config). These configuration files specify standardized parameters—such as sampling rate, prediction horizon, coordinate frames, and filter thresholds—to promote reproducibility and fair comparisons across experiments.
We have also included 10 Hz configurations for each dataset (different from the original 5 Hz configs), which is the dominant sampling rate for many instrumented-vehicle datasets.


## General Instructions
Before running any preprocessing script:

- Extract the raw dataset into a local directory (default: `../datasets`)
- Use the `--path` argument to specify the dataset location if different
- Use `--output-dir` to define where the preprocessed files should be saved (default: `./data`)
- Add the `--use-threads` flag to enable multithreaded processing (recommended)

Example:

```bash
python -m preprocessing.preprocess_urban --config rounD --path ../datasets --output-dir ./data --use-threads 1
```

Example using Apptainer:

```bash
apptainer run /path/to/dronalize.sif python -m preprocessing.preprocess_urban --config rounD --path ../datasets --output-dir ./data --use-threads 1
```

Example using Docker:

```bash
docker run \
  -v "$(pwd)":/app \
  -v "$(pwd)/../datasets":/datasets \
  -w /app \
  dronalize \
  python -m preprocessing.preprocess_urban \
    --config rounD \
    --path /datasets \
    --output-dir ./data \
    --use-threads 1
```