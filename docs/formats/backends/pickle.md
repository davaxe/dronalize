# Pickle

<div class="section-intro" markdown="1">
The `pickle` backend is the simplest persisted format in `dronalize`. It requires no optional
dependencies and stores one scene record per file.
</div>

## Why choose Pickle

Pickle is a good choice when you want:

- a dependency-light setup
- straightforward local inspection of per-scene files
- a simple baseline format for experimentation

Trade-offs:

- many small files for larger datasets
- not optimized for large-scale streaming training workflows
- no compression support

## Output directory structure

With a split strategy:

```text
output/
├── manifest.json
├── train/
│   ├── 000000.pkl
│   ├── 000001.pkl
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

Without a split strategy:

```text
output/
├── manifest.json
└── unsplit/
    ├── 000000.pkl
    ├── 000001.pkl
    └── ...
```

Each `.pkl` file contains one framework-neutral
[`SceneRecord`](../../reference/api/io/storage-and-manifests.md#dronalize.io.SceneRecord).

## Custom samples

Python integrations can customize what gets pickled by passing `output_sample`
to an
[`ExecutionRequest`](../../reference/api/runtime/planning-and-runs.md#dronalize.runtime.ExecutionRequest).

The preferred hook is `record_transform`, which receives the standard
`SceneRecord` after Dronalize has applied output schema conversion, precision,
recentering, map extraction, and default observation length metadata.

<!-- no-validate -->
```python
from dataclasses import dataclass

import numpy as np

from dronalize.io.records import SceneRecord
from dronalize.runtime import ExecutionRequest, OutputSample, execute_request


@dataclass(slots=True)
class TrainingSample:
    x: np.ndarray
    y: np.ndarray
    scene_number: int


def to_training_sample(record: SceneRecord) -> TrainingSample:
    observation_length = record.default_observation_length or 10
    return TrainingSample(
        x=record.features[:, :observation_length],
        y=record.features[:, observation_length:],
        scene_number=record.scene_number,
    )


request = ExecutionRequest(
    dataset="a43",
    input_dir=input_dir,
    output_dir=output_dir,
    storage_backend="pickle",
    output_sample=OutputSample(record_transform=to_training_sample),
)
execute_request(request)
```

The lower-level
[`PickleWriter`](../../reference/api/io/backends.md#dronalize.io.backends.pickle.PickleWriter)
constructor exposes the same capability as `record_transform` or
`scene_transform`.

<!-- no-validate -->
```python
writer = PickleWriter(
    output_dir=output_dir, config=output_plan, splits=None, record_transform=to_training_sample
)
```

For advanced cases, `scene_transform` can derive the persisted object directly
from the runtime `Scene`. This bypasses `SceneRecord` encoding, so the transform
is responsible for any schema conversion, dtype policy, recentering, and map
resolution it needs. `record_transform` and `scene_transform` are mutually
exclusive.

## Reading Pickle output

<!-- no-validate -->
```python
from pathlib import Path
from dronalize.io.readers import PickleReader

reader = PickleReader(Path("output"), split="train")
record = reader[0]

print(record.scene_number)
print(record.dataset)
print(record.features.shape, record.mask.shape)
```

Use `split=None` (default) for unsplit exports.

For custom pickled samples, pass the expected sample type:

<!-- no-validate -->
```python
reader = PickleReader(Path("output"), sample_type=TrainingSample)
sample = reader[0]
```

## Configuration

Pickle has no backend-specific config block. It uses the general output settings:

- schema
- precision
- recentering

See [Outputs and schemas](../schema.md) for details.
