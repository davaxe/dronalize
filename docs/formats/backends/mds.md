# Mosaic data shard (MDS)

<div class="section-intro" markdown="1">
MDS is an optional storage backend for `dronalize`. It writes processed scenes as binary shards
readable by the [Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/index.html)
library, which supports efficient streaming and shuffling during model training without requiring the
full dataset to fit in memory.
</div>

!!! warning "MDS requires additional dependencies"
    To use MDS export and reading features, you must install the `mds` extra:

    ```bash
    pip install dronalize[mds]
    ```

## Why choose MDS

MDS is a strong fit when you need:

- sharded binary output for large datasets
- configurable compression and shard sizing
- efficient integration with streaming-style training pipelines
- consistent reader behavior across local and remote storage setups

Trade-offs:

- requires the optional `mds` dependency set
- output is binary and less inspectable than per-scene pickle files

## Output directory structure

Each processing run writes output under the directory provided by `--output`. The exact layout
depends on whether a split strategy is active.

With a split strategy, subdirectories are created for each emitted split (not every run produces all three):

```text
output/
├── train/
│   ├── index.json
│   └── shard.00000.mds
├── val/
│   ├── index.json
│   └── shard.00000.mds
└── test/
    ├── index.json
    └── shard.00000.mds
```

Without a split strategy:

```text
output/
├── manifest.json
└── unsplit/
    ├── index.json
    └── shard.00000.mds
```

`manifest.json` at the output root is written by `dronalize` and is common across backends.
`index.json` and `shard.*.mds` are the MDS split payload.

Multiple shard files are created automatically as a split exceeds `size_limit`.

!!! note "Parallel execution"
    When `jobs > 1`, each worker writes to a temporary subdirectory inside the split directory.
    After all workers finish, `dronalize` merges the per-worker shard indexes into a single
    `index.json` at the split root. The final layout will include subdirectories for each
    worker, but will work as expected because the merged index points to those shards.
    
    This parallel approach is described in more detail in the [parallel dataset conversion guide](https://docs.mosaicml.com/projects/streaming/en/stable/preparing_datasets/parallel_dataset_conversion.html).


## Sample fields

Every sample written to a shard contains the following fields.

| Field | dtype | Shape | Description |
| --- | --- | --- | --- |
| `scene_number` | `int` | scalar | Global scene index assigned during processing. |
| `dataset` | `str` | scalar | Dataset label associated with the sample. |
| `default_observation_length` | `int` | scalar | Default split point for input/output windows during reading, or `-1` when unset. |
| `position_offset` | `float64` | `[2]` | The `(x, y)` offset subtracted from all positions before writing when `recenter_positions = true`. Zero otherwise. |
| `agent_types` | `int32` | `[A]` | Integer agent category for each agent. |
| `screened_agent_mask` | `uint8` | `[A]` | Per-agent screening pass mask (`1` means passed). |
| `features` | `float32` or `float64` | `[A, T, F]` | Per-agent feature tensor in canonical column order. |
| `mask` | `uint8` | `[A, T]` | Presence mask. `1` where the agent is observed at that timestep, `0` otherwise. |
| `map_node_positions` | `float32` or `float64` | `[N, 2]` | Map lane node coordinates, offset-corrected when applicable. |
| `map_edge_indices` | `int32` | `[2, E]` | Edge connectivity in COO format (source and target node indices). |
| `map_node_types` | `int32` | `[N]` | Integer node type per map node. |
| `map_edge_types` | `int32` | `[E]` | Integer edge type per map edge. |

**Dimension key:**

- `A` — number of agents in the scene (varies per sample)
- `T` — total full-horizon timesteps (fixed for a given dataset and config)
- `F` — number of feature columns determined by the configured schema
- `N` — number of map nodes (varies per sample)
- `E` — number of map edges (varies per sample)

The `features` and `map_node_positions` dtypes follow the `precision` setting in the `[output]`
config block. `position_offset` is always stored as `float64`.

!!! note "Map fields when maps are disabled"
    All four map fields are always present in every sample regardless of whether map data is
    enabled. For scenes without maps, encoded placeholders are normalized back to empty arrays by
    the reader API.
    
!!! warning "Map fields will not be empty arrays"
    When maps are disabled, the map fields are still present but contain encoded placeholder values.
    The reader API normalizes these back to empty arrays, but they will not be empty in the raw
    MDS shards.
    
    The reason for this design is that the MDS format does not currently support empty arrays.

## Custom samples

Python integrations can customize the sample dictionaries written to MDS by
passing `output_sample` to an
[`ExecutionRequest`](../../reference/api/runtime/planning-and-runs.md#dronalize.runtime.ExecutionRequest).

MDS requires the column schema before any samples are written, so custom
transforms must provide both:

- `record_transform` or `scene_transform`
- `mds_columns`

The preferred hook is `record_transform`, which receives the standard
`SceneRecord` after Dronalize has applied output schema conversion, precision,
recentering, map extraction, and default observation length metadata.

<!-- no-validate -->
```python
import numpy as np

from dronalize.io.records import SceneRecord
from dronalize.runtime import ExecutionRequest, OutputSample, execute_request


def to_training_sample(record: SceneRecord) -> dict[str, object]:
    observation_length = record.default_observation_length or 10
    return {
        "scene_number": int(record.scene_number),
        "x": record.features[:, :observation_length],
        "y": record.features[:, observation_length:],
        "x_mask": record.mask[:, :observation_length].astype(np.uint8, copy=False),
        "y_mask": record.mask[:, observation_length:].astype(np.uint8, copy=False),
    }


columns = {
    "scene_number": "int",
    "x": "ndarray:float32",
    "y": "ndarray:float32",
    "x_mask": "ndarray:uint8",
    "y_mask": "ndarray:uint8",
}

request = ExecutionRequest(
    dataset="a43",
    input_dir=input_dir,
    output_dir=output_dir,
    storage_backend="mds",
    output_sample=OutputSample(record_transform=to_training_sample, mds_columns=columns),
)
execute_request(request)
```

The lower-level
[`MDSDatasetWriter`](../../reference/api/io/backends.md#dronalize.io.backends.mds.MDSDatasetWriter)
constructor exposes the same capability as `record_transform` or
`scene_transform` plus `sample_columns`.

<!-- no-validate -->
```python
writer = MDSDatasetWriter(
    output_dir=output_dir,
    config=output_plan,
    splits=None,
    parallel=False,
    record_transform=to_training_sample,
    sample_columns=columns,
)
```

For advanced cases, `scene_transform` can derive the MDS sample directly from
the runtime `Scene`. This bypasses `SceneRecord` encoding, so the transform is
responsible for any schema conversion, dtype policy, recentering, and map
resolution it needs. `record_transform` and `scene_transform` are mutually
exclusive.

Custom MDS outputs should be read with `convert_raw`:

<!-- no-validate -->
```python
from dataclasses import dataclass

from dronalize.io.readers import MDSReader


@dataclass(slots=True)
class TrainingSample:
    x: np.ndarray
    y: np.ndarray
    x_mask: np.ndarray
    y_mask: np.ndarray


def from_raw(sample: dict[str, object]) -> TrainingSample:
    return TrainingSample(
        x=np.asarray(sample["x"]),
        y=np.asarray(sample["y"]),
        x_mask=np.asarray(sample["x_mask"], dtype=bool),
        y_mask=np.asarray(sample["y_mask"], dtype=bool),
    )


reader = MDSReader(path=output_dir, convert_raw=from_raw)
```

## Configuration

MDS-specific export settings are controlled under `[output.mds]` in the config file.
The main `[output]` block controls schema, precision, and position offsetting.

For the full field reference see the [output configuration](../../reference/configuration/output.md)
page.

A typical setup:

```toml
[datasets.a43.output]
schema = "positions_velocity_yaw"
precision = "float32"
recenter_positions = true

[datasets.a43.output.mds]
compression = "zstd:7"
size_limit = 67108864
exist_ok = false
```

## How to read produced datasets

See [Reading data](../reading.md) for
[`MDSReader`](../../reference/api/io/readers.md#dronalize.io.readers.MDSReader) usage and optional Torch/PyG
adapters.

*[COO]: Edge format for sparse graphs compatible with PyTorch Geometric.
*[MDS]: Mosaic Data Shard, a binary data format developed by MosaicML for efficient streaming and shuffling during model training.
