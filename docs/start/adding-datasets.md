# Adding datasets

<div class="section-intro" markdown="1">
This page is a focused guide for advanced users who want to add a custom dataset to
`dronalize`. The stable extension surface is the dataset registry plus the loader API
under `dronalize.processing.loading`.
</div>

## Start with the dataset contract

Before writing code, pin down the dataset contract:

- what the raw sources are: files, directories, archives, or database records
- whether the data is already split into scenes, recordings, or sequences
- which trajectory fields are available
- whether timestamps are frame-based, wall-clock-based, or irregular
- whether maps, agent categories, or split annotations are included

These answers determine the loader shape, default config, schema, and optional capabilities.

## Implement a loader

Subclass `BaseSceneLoader` and implement the dataset-specific pieces only:

- default loader config
- native trajectory schema
- source discovery
- source loading
- scene preparation
- optional map handling
- optional native split support

Import from:

```python
from dronalize.processing.loading import BaseSceneLoader
```

The loader should normalize raw records into the shared scene representation and let
the common processing pipeline handle filtering, resampling, split assignment, and export.

## Define and register a descriptor

Wrap the loader in a `DatasetSpec` and register it:

```python
from dronalize.datasets import DatasetSpec, register

register(DatasetSpec.from_loader("my_dataset", MyLoader, infer_capabilities=True))
```

`DatasetSpec` is the public contract that exposes defaults, supported split strategies,
map support, loader options, and optional runtime context.

## Add optional capabilities deliberately

Start with the minimal integration that produces correct scenes. Add advanced support
only when the dataset actually has it and the loader implementation is clear:

- map or lane-graph support
- native train/val/test splits
- dataset-specific loader options
- runtime context for shared resources

## Keep the split of responsibilities clean

Put dataset-specific concerns in the loader:

- source layout
- parsing and normalization
- coordinate conventions
- dataset-specific metadata

Leave shared concerns to the common runtime:

- config merging
- split-policy wiring
- filtering and resampling
- export layout

## See also

- [Loader extension API](../reference/api/processing/loading.md)
- [Dataset specs and capabilities](../reference/api/datasets/descriptors.md)
- [Architecture](../concepts/architecture.md)
