# Mosaic data shard (MDS)

<div class="section-intro" markdown="1">
MDS is the primary storage backend for `dronalize`. It writes processed scenes as binary shards
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
    `index.json` at the split root. The final layout is identical to the serial case.

## Sample fields

Every sample written to a shard contains the following fields.

| Field | dtype | Shape | Description |
| --- | --- | --- | --- |
| `scene_number` | `int` | scalar | Global scene index assigned during processing. |
| `observation_length` | `int` | scalar | Number of history frames used to split features into input/output windows during reading. |
| `position_offset` | `float64` | `[2]` | The `(x, y)` offset subtracted from all positions before writing when `recenter_positions = true`. Zero otherwise. |
| `agent_types` | `int32` | `[A]` | Integer agent category for each agent. |
| `passed_agent_mask` | `uint8` | `[A]` | Per-agent screening pass mask (`1` means passed). |
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
    enabled. For scenes without maps, encoded placeholders are normalized back to empty arrays by
    the reader API.

## Configuration

MDS-specific export settings are controlled under `[export.backends.mds]` in the config file.
The main `[export]` block controls schema, precision, and position offsetting.

For the full field reference see the [output configuration](../../reference/configuration/output.md)
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

See [Reading data](../reading.md) for `MDSReader` usage and optional Torch/PyG adapters.

*[COO]: Edge format for sparse graphs compatible with PyTorch Geometric.
*[MDS]: Mosaic Data Shard, a binary data format developed by MosaicML for efficient streaming and shuffling during model training.
