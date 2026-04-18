# Adding datasets

<div class="section-intro" markdown="1">
The stable extension surface for new datasets is the dataset registry plus the loader API in
[`dronalize.processing.loading`][dronalize.processing.loading]. The goal of a
dataset integration is to turn raw sources into the shared scene model and let
the common runtime handle screening, splitting, schema conversion, and writing.
</div>

## Start with the dataset contract

Before writing code, pin down:

- what counts as one raw source: file, directory, archive member, or database row
- whether the dataset already has native train, val, or test partitions
- which trajectory fields are present in the native data
- whether timestamps are regular enough for windowing and resampling
- whether map data exists, and if it is scene-local or shared across many scenes
- whether the dataset needs custom options under `[datasets.<name>.dataset]`

These answers determine the loader shape, the
[`DatasetSpec`][dronalize.datasets.DatasetSpec], and the default config you
should publish.

## Implement a loader

Subclass [`BaseSceneLoader`][dronalize.processing.loading.BaseSceneLoader] and
implement the dataset-specific pieces only.

```python
from dronalize.processing.loading import BaseSceneLoader, LoadedSourceData, Source
```

In practice, a loader needs to:

- implement `native_trajectory_schema()`
- implement `load_source()`
- implement either `discover_sources()` or `sources_for_split()`

Most loaders can inherit the default pipeline behavior from
[`BaseSceneLoader`][dronalize.processing.loading.BaseSceneLoader]. That gives
you the standard `scenes`, `screening`, `split`, `map`, and lane-change
handling automatically.

If your dataset needs dataset-owned config, define a typed options model and let the loader read it
from `self.dataset_config`. Those values come from `[datasets.<name>.dataset]`.

## Register a [`DatasetSpec`][dronalize.datasets.DatasetSpec]

Wrap the loader in a [`DatasetSpec`][dronalize.datasets.DatasetSpec] and
register it:

```python
from dronalize.config.models import DatasetConfig, ScenesConfig
from dronalize.datasets import DatasetSpec, register

register(
    DatasetSpec(
        name="my_dataset",
        loader_factory=MyLoader.unified_factory,
        default_config=DatasetConfig(
            scenes=ScenesConfig(history_frames=20, future_frames=30, sample_time=0.1)
        ),
        native_schema=MyLoader.native_trajectory_schema(),
    )
)
```

Add the other [`DatasetSpec`][dronalize.datasets.DatasetSpec] fields only when
the dataset actually supports them:

- `native_splits` when the raw dataset ships with fixed partitions
- `has_map` when map data is available
- `dataset_options_model` when `[datasets.<name>.dataset]` should be typed and validated
- `resources_factory` when a run should build or cache shared resources such as maps once
- `time_split_support` when time-based split strategies are valid for the dataset

## Keep responsibilities clean

Put dataset-specific logic in the loader:

- source discovery
- raw parsing and normalization
- coordinate conventions
- dataset-specific metadata
- map lookup details

Leave shared runtime behavior to the library:

- config layering and override handling
- screening and resampling
- split assignment
- schema conversion
- backend writing and manifest creation

## Practical guidance

- Start with the smallest correct integration.
- Reuse the common pipeline unless the dataset genuinely needs a custom one.
- Add map support only when you can resolve maps reliably for each scene.
- Add native split support only when the dataset really defines stable partitions.
- Use dataset-owned config only for options that do not belong in the shared public config model.

## See also

- [Architecture](../concepts/architecture.md)
- [Datasets](../concepts/datasets.md)
- [Configuration model](../concepts/configuration-model.md)
