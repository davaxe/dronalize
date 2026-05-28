# Reading data

<div class="section-intro" markdown="1">
`dronalize` readers expose one common in-memory record model across storage backends. This means
you can switch between `pickle` and `mds` without changing your downstream scene-processing code.
</div>

## Reader model

Framework-neutral readers return [`SceneRecord`](../reference/api/io/index.md#dronalize.io.SceneRecord)
objects with:

- scene id and position offset
- full-horizon agent features and masks
- optional map graph arrays

Use [`SceneRecord.split`](../reference/api/io/index.md#dronalize.io.SceneRecord.split)
when a model needs explicit observation and prediction tensors.

Use the same post-processing logic regardless of backend.

## Read the manifest first

<!-- no-validate -->
```python
from pathlib import Path
from dronalize.io import read_manifest

manifest = read_manifest(Path("output"))
print(manifest.feature_columns)
print(manifest.trajectory_schema_fields)
print(manifest.horizon_frames, manifest.default_observation_length)
```

Reading the manifest up front with
[`read_manifest()`](../reference/api/io/index.md#dronalize.io.read_manifest) is the easiest way to verify
schema, horizon, and precision.

!!! tip "Readable manifest"
    The manifest is stored in a human-readable JSON format, so you can also
    open and inspect it manually if needed.


## Read from Pickle output

<!-- no-validate -->
```python
from pathlib import Path
from dronalize.io.readers import PickleReader

reader = PickleReader(Path("output"), split="train")

print(len(reader))
scene = reader[0]
print(scene.features.shape, scene.mask.shape)

if manifest.default_observation_length is not None:
    split = scene.split(manifest.default_observation_length)
    print(split.history_features.shape, split.future_features.shape)
```

For unsplit exports, use `split=None` (the default), which reads from `unsplit/`.

## Read from MDS output

!!! warning "MDS requires extra dependencies"
    Install the MDS extra before using MDS readers: `pip install dronalize[mds]`.

<!-- no-validate -->
```python
from pathlib import Path
from dronalize.io.readers import MDSReader

reader = MDSReader(path=Path("output"), split="train")

print(len(reader))
scene = reader[0]
print(scene.features.shape, scene.mask.shape)
```

For unsplit exports, use `split=None` (default), which reads from `unsplit/`.

## Torch and PyG adapters

On top of the readers, `dronalize` provides optional adapters:

- [`TorchSceneDataset`](../reference/api/io/adapters.md#dronalize.io.adapters.TorchSceneDataset) for full-horizon
  Torch tensor records
- [`TorchSplitSceneDataset`](../reference/api/io/adapters.md#dronalize.io.adapters.TorchSplitSceneDataset) for Torch
  tensor records split at an explicit `observation_length`, a callable resolver, or the per-record
  `default_observation_length`
- [`HeteroSceneDataset`](../reference/api/io/adapters.md#dronalize.io.adapters.HeteroSceneDataset) for full-horizon
  PyTorch Geometric `HeteroData`
- [`SplitHeteroSceneDataset`](../reference/api/io/adapters.md#dronalize.io.adapters.SplitHeteroSceneDataset)
  for PyTorch Geometric `HeteroData` split at an explicit `observation_length`, a callable resolver,
  or the per-record `default_observation_length`

Use these when your training stack expects framework-native dataset objects.

## Choosing a reader setup

- Use [`PickleReader`](../reference/api/io/readers.md#dronalize.io.readers.PickleReader) for simple local
  workflows and easy inspection.
- Use [`MDSReader`](../reference/api/io/readers.md#dronalize.io.readers.MDSReader) for larger-scale or
  streaming-oriented training pipelines.
- Keep reader-side code backend-neutral by depending on the shared
  [`SceneRecord`](../reference/api/io/index.md#dronalize.io.SceneRecord) contract.
