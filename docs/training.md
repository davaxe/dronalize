# Model Training
This page provides an overview of how to train models using the Dronalize toolbox, including the training script and
configuration options.

## Overview
The toolbox includes a training script, [train.py](train.py), that can be used to train your models on the preprocessed
data.
The script is designed to be run from the repository root directory and includes several arguments that can be used to
configure the training process.

## Integration with Lightning
The training script is designed to be used with PyTorch Lightning; besides using the custom data modules described in [Data Loading](docs/data_loading.md), it also requires a `LightningModule` that defines the model and training loop.
In [models/prototype/litmodule.py](models/prototype/litmodule.py), you will find a base class that can be modified to
build your own `LightningModule`.
In its current form, it can be used to train and evaluate the baseline model.
It also details how to use the proposed evaluation metrics for trajectory prediction.

## Configuration
By default, it uses configuration files in `.yml` format found in the [configs](configs) directory, detailing the
required modules and hyperparameters for training.
Additional runtime arguments, such as the number of workers, GPU acceleration, debug mode, and model checkpointing, can
be specified when running the script (see [arguments.py](arguments.py) for more information).

Example command to run the training script:

```bash
  python train.py --add-name Test --dry-run 0 --use-cuda 1 --num-workers 4
```

Example using Apptainer:

```bash
apptainer run --nv build/dronalize.sif python train.py -an Test -dr 0 -uc 1 -nw 4
``` 

Example using Docker:

```bash
  docker run --gpus all -v "$(pwd)":/app -w /app dronalize python train.py
  -an Test -dr 0 -uc 1 -nw 4
```

We recommend users modify the default arguments in [arguments.py](arguments.py) to suit their needs.


## Logging
Note that the default logger is set to `wandb` ([weights & biases](https://wandb.ai/)) for logging performance metrics
during training.
It is our preferred tool for tracking experiments, but it can be easily replaced with other logging tools by modifying
the `Trainer` in the training script.

See the official [Lightning documentation](https://lightning.ai/docs/pytorch/stable/) for more information on
customizing training behavior and how to use the library in general.
