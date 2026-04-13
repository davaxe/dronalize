# Reading data

<div class="section-intro" markdown="1">
`dronalize` readers expose one common in-memory record model across storage backends. This means
you can switch between `pickle` and `mds` without changing your downstream scene-processing code.
</div>

## Reader model

Framework-neutral readers return `SceneRecord` objects with:

- scene id and position offset
- agent features and masks split into history and future windows
- optional map graph arrays

Use the same post-processing logic regardless of backend.

## Read the manifest first

```python
from pathlib import Path
from dronalize.io import read_manifest

manifest = read_manifest(Path("output"))
print(manifest.feature_columns)
print(manifest.history_frames, manifest.future_frames)
```

Reading the manifest up front is the easiest way to verify schema, horizon, and precision.

## Read from Pickle output

```python
from pathlib import Path
from dronalize.io.readers import PickleReader

reader = PickleReader(Path("output"), split="train")

print(len(reader))
scene = reader[0]
print(scene.input_features.shape, scene.output_features.shape)
```

For unsplit exports, use `split=None` (the default), which reads from `unsplit/`.

## Read from MDS output

!!! warning "MDS requires extra dependencies"
    Install the MDS extra before using MDS readers: `pip install dronalize[mds]`.

```python
from pathlib import Path
from dronalize.io.readers import MDSReader

reader = MDSReader(path=Path("output"), split="train")

print(len(reader))
scene = reader[0]
print(scene.input_features.shape, scene.output_features.shape)
```

For unsplit exports, use `split=None` (default), which reads from `unsplit/`.

## Torch and PyG adapters

On top of the readers, `dronalize` provides optional adapters:

- `TorchSceneDataset` for Torch tensor records
- `HeteroSceneDataset` for PyTorch Geometric `HeteroData`

Use these when your training stack expects framework-native dataset objects.

## Choosing a reader setup

- Use `PickleReader` for simple local workflows and easy inspection.
- Use `MDSReader` for larger-scale or streaming-oriented training pipelines.
- Keep reader-side code backend-neutral by depending on the shared `SceneRecord` contract.
