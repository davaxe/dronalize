# Loading Data
This page provides an overview of how to load preprocessed trajectory data into PyTorch training pipelines using the Dronalize toolbox.

### Datamodules
In [datamodules](datamodules), you will find the necessary classes for loading the preprocessed data into PyTorch
training pipelines.
It includes:

- `TrajDataset`: A `Dataset` class built around `torch_geometric`. Found in: [dataset.py](datamodules/dataset.py)
- `TrajDataModule`: A `DataModule` class, including `Dataloader` built around `lightning.pytorch`. Found
  in: [dataloader.py](datamodules/dataloader.py)
- `CoordinateTransform` and `CoordinateShift`: Example transformations. Found
  in: [transforms.py](datamodules/transforms.py)

> [dataloader.py](datamodules/dataloader.py) is designed to be runnable as a standalone script for quick testing of the
> data loading pipeline.
> It includes a `main` function that can be used to load the data and visualize it for debugging and/or educational
> purposes.