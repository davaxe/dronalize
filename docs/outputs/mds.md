# Mosaic Data Shard (MDS)

<div class="section-intro" markdown="1">
MDS is the primary storage backend for `dronalize`. It writes processed scenes as binary shards
readable by the [Mosaic Streaming](https://docs.mosaicml.com/projects/streaming/en/stable/index.html)
library, which supports efficient streaming and shuffling during model training without requiring the
full dataset to fit in memory.
</div>

## Why MDS

MDS allow some conviences for writing and reading. In particular, on the format supports:

-  First class support for dynamic array shapes, which makes it easy to write data with variable number of agents and map elements per scene.
-  Built-in compression and sharding, which can be configured to optimize for storage and reading speed.
-  Support for merging dataset by leveraging the `Stream` abstraction.
-  Design for distributed training and supports non-local storage backends, which are downloaded on demand during training.

Some downsides to be aware of:

-  `torch` dependency for writing and reading.
-  Not human-readable on disk, which can make debugging more difficult.
-  Seemingly not that widely adotped, and as of writing the activity on the Github repo is low.

## Output directory structure

Each processing run writes output under the directory provided by `--output`. The exact layout
depends on whether a split strategy is active.

With a split strategy subdirectories are created for each split (not all splits are always present, and depends on the configured strategy):

```text
output/
├── train/
│   ├── manifest.json
│   ├── index.json
│   └── shard.00000.mds
├── val/
│   ├── manifest.json
│   ├── index.json
│   └── shard.00000.mds
└── test/
    ├── manifest.json
    ├── index.json
    └── shard.00000.mds
```

Without a split strategy:

```text
output/
└── all/
    ├── manifest.json
    ├── index.json
    └── shard.00000.mds
```

`manifest.json` is written by `dronalize` and is common across all storage backends — see the
[storage backends page](index.md#manifest) for its contents and how to read it. `index.json` and the
`shard.*.mds` files are written by the MDS library itself. Multiple shard files are created when a
split's data exceeds the configured `size_limit`.

!!! note "Parallel execution"
    When `jobs > 1`, each worker writes to a temporary subdirectory inside the split directory.
    After all workers finish, `dronalize` merges the per-worker shard indexes into a single
    `index.json` at the split root. The final layout is identical to the serial case.

## Sample fields

Every sample written to a shard contains the following fields.

| Field | dtype | Shape | Description |
| --- | --- | --- | --- |
| `scene_number` | `int` | scalar | Global scene index assigned during processing. |
| `history_frames` | `int` | scalar | Number of history frames in this scene. |
| `future_frames` | `int` | scalar | Number of future frames in this scene. |
| `position_offset` | `float64` | `[2]` | The `(x, y)` offset subtracted from all positions before writing when `recenter_positions = true`. Zero otherwise. |
| `agent_types` | `int32` | `[A]` | Integer agent category for each agent. |
| `features` | `float32` or `float64` | `[A, T, F]` | Per-agent feature tensor in canonical column order. |
| `mask` | `uint8` | `[A, T]` | Presence mask. `1` where the agent is observed at that timestep, `0` otherwise. |
| `map_node_positions` | `float32` or `float64` | `[N, 2]` | Map lane node coordinates, offset-corrected when applicable. |
| `map_edge_indices` | `int32` | `[2, E]` | Edge connectivity in COO format (source and target node indices). |
| `map_node_types` | `int32` | `[N]` | Integer node type per map node. |
| `map_edge_types` | `int32` | `[E]` | Integer edge type per map edge. |

**Dimension key:**

- `A` — number of agents in the scene (varies per sample)
- `T` — total timesteps: `history_frames + future_frames` (fixed for a given dataset and config)
- `F` — number of feature columns determined by the configured schema
- `N` — number of map nodes (varies per sample)
- `E` — number of map edges (varies per sample)

The `features` and `position_offset` dtypes follow the `precision` setting in the `[export]` config
block. Map fields match the same precision.

!!! note "Map fields when maps are disabled"
    All four map fields are always present in every sample regardless of whether map data is
    enabled. When a scene has no map — either because the dataset does not support maps or because
    `enabled = false` — the map fields are filled with sentinel values: `NaN` positions,
    `−1` node and edge types, and a single dummy node and edge.

When read back through `dronalize.io.readers.mds.MDSReader`, those sentinel values are converted
back into empty map arrays. `MDSTorchDataset` and `MDSHeteroDataset` then build on top of the
same normalized scene records.

## Configuration

MDS-specific export settings are controlled under `[export.backends.mds]` in the config file. The
main `[export]` block controls schema, precision, and position offsetting.

For the full field reference see the [export configuration](../reference/configuration/export.md)
page.

A typical setup:

```toml
[datasets.a43.export]
schema = "positions_velocity_yaw"
precision = "float32"
recenter_positions = true

[datasets.a43.export.backends.mds]
compression = "zstd:7"
size_limit = 67108864
exist_ok = false
```


## How to read produced datasets

Use `MDSReader` when you want framework-neutral scene records backed by NumPy arrays:

```python
from pathlib import Path

from dronalize.io.readers.mds import MDSReader

reader = MDSReader(path=Path("output"), split="train")
record = reader[0]

print(record.input_features.shape)
print(record.map_edge_indices.shape)
```

`MDSReader` is a thin wrapper around Mosaic Streaming's `StreamingDataset`, but it exposes a
framework-neutral read surface. The returned `RawSceneRecord` objects are summarized by the field
table above. The `split` argument is optional, but if it is omitted the path must point to the
split subdirectory, for example `"output/train"` instead of just `"output"`.

Use `MDSTorchDataset` when you want the same records converted into Torch tensors through an
iterable dataset surface:

```python
from pathlib import Path

from dronalize.io.adapters import MDSTorchDataset

dataset = MDSTorchDataset(path=Path("output"), split="train")
sample = dataset[0]

print(sample.input_features.shape)
print(sample.map_edge_indices.shape)
```

`MDSTorchDataset` sets `batch_size=1` by default when it constructs its own `MDSReader`, because
Mosaic Streaming requires a batch size for iterable access. When using it with a `DataLoader`,
pass `batch_size=...` to `MDSTorchDataset` so the reader and loader agree.

!!! note "Using `StreamingDataset`"
    If needed the
    [`StreamingDataset`](https://docs.mosaicml.com/projects/streaming/en/stable/api_reference/generated/streaming.StreamingDataset.html#streamingdataset)
    (from the Mosaic Streaming library) can be used directly.

If a PyTorch Geometric compatible dataset is needed, install the `pyg` extra and use
`MDSHeteroDataset`:

```python
from pathlib import Path

from dronalize.io.adapters import MDSHeteroDataset

dataset = MDSHeteroDataset(path=Path("output"), split="train")
graph = dataset[0]

print(graph["agent"].x.shape)
print(graph["map", "connects", "map"].edge_index.shape)
```

If batch-local time padding is needed for variable-length scenes, use
`collate_hetero_with_time_padding` as the `DataLoader` collate function:

```python
from torch.utils.data import DataLoader

from dronalize.io.adapters import MDSHeteroDataset, collate_hetero_with_time_padding

dataset = MDSHeteroDataset(path=Path("output"), split="train")
loader = DataLoader(dataset, batch_size=8, collate_fn=collate_hetero_with_time_padding)

batch = next(iter(loader))
print(batch["agent"].x.shape)
```

The collator keeps the agent and map dimensions dynamic across scenes, and pads only the input and
output time axes to the maximum lengths present in the current batch.

`MDSHeteroDataset` builds on top of `MDSTorchDataset`, and both wrap `MDSReader`, so all three
surfaces interpret the on-disk MDS scene records the same way.

*[COO]: Edge format for sparse graphs compatible with PyTorch Geometric.
*[MDS]: Mosaic Data Shard, a binary data format developed by MosaicML for efficient streaming and shuffling during model training.
