# Manifest

<div class="section-intro" markdown="1">
The manifest is the stable human readable metadata contract for a processed dataset. It records how scenes were
produced so downstream readers can interpret feature tensors correctly without inspecting scene
files directly.
</div>

## Location

`dronalize` writes one `manifest.json` at the output root directory. The parsed
Python representation is [`DatasetManifest`](../reference/api/io/storage-and-manifests.md#dronalize.io.DatasetManifest).

Typical layout:

```text
output/
├── manifest.json
├── train/
│   └── ... backend-specific scene files ...
├── val/
│   └── ...
└── test/
    └── ...
```

When no split strategy is active, scene data is written to `unsplit/`.

## Manifest fields

| Field | Type | Description |
| --- | --- | --- |
| `format_version` | `int` | Manifest format version used for compatibility checks. |
| `source_trajectory_schema` | `str` | Native schema of the source dataset before conversion. |
| `trajectory_schema` | `str` | Schema written to output. |
| `derived_features` | `list[str]` | Features derived during schema conversion instead of read directly from source data. |
| `feature_columns` | `list[str]` | Ordered feature column names in the persisted tensors. |
| `history_frames` | `int` | Number of history timesteps in each scene. |
| `future_frames` | `int` | Number of future timesteps in each scene. |
| `precision` | `str` | Floating-point precision used for persisted features (`"float32"` or `"float64"`). |
| `recenter_positions` | `bool` | Whether per-scene position recentering was applied. |
| `has_map` | `bool` | Whether map data is included in the output. |
| `sample_time` | `float` | Effective sample interval (seconds) after any resampling. |
| `original_sample_time` | `float` | Source sample interval (seconds) before resampling. |

## Why it matters

The manifest enables robust downstream loading by making key assumptions explicit:

- feature order is known (`feature_columns`)
- scene horizon is known (`history_frames`, `future_frames`)
- numeric and coordinate conventions are known (`precision`, `recenter_positions`)
- map availability is explicit (`has_map`)

## Read it programmatically

```python
from pathlib import Path
from dronalize.io import read_manifest

manifest = read_manifest(Path("output"))

print(manifest.trajectory_schema)
print(manifest.feature_columns)
print(manifest.history_frames, manifest.future_frames)
```

See [`read_manifest()`](../reference/api/io/storage-and-manifests.md#dronalize.io.read_manifest) for the full reader API.
