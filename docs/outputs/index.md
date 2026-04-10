# Storage Backends

<div class="section-intro" markdown="1">
`dronalize` is designed to support multiple storage backends. The only fully supported backend today is
*Mosaic Data Shard (MDS)*, developed by [MosaicML](https://docs.mosaicml.com/projects/streaming/en/stable/index.html).
A `null` backend is also available for validation and debugging purposes.
</div>

## Selecting a backend

The storage backend is a run-level decision set with `--storage-backend` on the CLI. It cannot be
specified in the TOML config file. The default when the flag is omitted is `mds`.

| Backend | Purpose |
| --- | --- |
| `mds` | Production runs. Writes binary shards and a shard index readable by the Mosaic Streaming library. |
| `null` | Debugging and config validation. Runs the full pipeline without writing any files to disk. |

The `null` backend is useful when you want to verify that a config is valid, check how many scenes
a dataset produces, or benchmark the pipeline itself without the cost of I/O.

See the [MDS backend](mds.md) page for output structure, scene fields, and configuration options.

## Manifest

Regardless of storage backend, every split directory produced by `dronalize` contains a
`manifest.json`. This file is written by `dronalize` and records the processing decisions made for that run.

It serves as the stable metadata contract between a preprocessing run and any downstream data
loader: everything needed to correctly interpret the written scenes is present in the manifest
without having to read the scene records themselves.

| Field | Type | Description |
| --- | --- | --- |
| `format_version` | `int` | Schema version for forward-compatibility checks. |
| `source_trajectory_schema` | `str` | Native schema of the source dataset before any conversion. |
| `trajectory_schema` | `str` | Schema actually written to disk. |
| `derived_features` | `list[str]` | Feature columns derived during schema conversion rather than read directly from the source. |
| `feature_columns` | `list[str]` | Ordered list of column names in the feature tensor. |
| `history_frames` | `int` | Number of history frames. |
| `future_frames` | `int` | Number of future frames. |
| `precision` | `str` | Floating-point precision: `"float32"` or `"float64"`. |
| `recenter_positions` | `bool` | Whether positions were recentered before writing. |
| `has_map` | `bool` | Whether map data is present in the output. |
| `sample_time` | `float` | Seconds per frame after any resampling. |
| `original_sample_time` | `float` | Seconds per frame of the original source data before resampling. |

The manifest can be read programmatically with `dronalize.io.read_manifest`:

```python
from pathlib import Path
from dronalize.io import read_manifest

manifest = read_manifest(Path("output/train"))
print(manifest.feature_columns)  # ('x', 'y', 'vx', 'vy', 'yaw')
print(manifest.history_frames, manifest.future_frames)
print(manifest.has_map)
```
