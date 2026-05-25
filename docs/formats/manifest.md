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

Current manifests use `format_version = 2`.

| Field | Type | Description |
| --- | --- | --- |
| `dataset` | `str` | Name of the dataset. |
| `storage_backend` | `str` | Storage backend used to write scene records. |
| `dronalize_version` | `str` | Package version that produced the export. |
| `format_version` | `int` | Manifest format version used for compatibility checks. |
| `source_trajectory_schema` | `str` | Native schema of the source dataset before conversion. |
| `source_trajectory_schema_fields` | `list[str]` | Ordered semantic fields in the native schema. |
| `trajectory_schema` | `str` | Schema written to output. |
| `trajectory_schema_fields` | `list[str]` | Ordered semantic fields in the exported schema. |
| `derived_features` | `list[str]` | Features derived during schema conversion instead of read directly from source data. |
| `feature_columns` | `list[str]` | Ordered feature column names in the persisted tensors. |
| `horizon_frames` | `int` | Number of full-horizon timesteps in each persisted scene. |
| `default_observation_length` | `int` or `null` | Default split point for reader-side observation/prediction views. |
| `precision` | `str` | Floating-point precision used for persisted features (`"float32"` or `"float64"`). |
| `recenter_positions` | `bool` | Whether per-scene position recentering was applied. |
| `has_map` | `bool` | Whether the run requested map output and records may contain map topology arrays. |
| `sample_time` | `float` | Effective sample interval (seconds) after any resampling. |
| `original_sample_time` | `float` | Source sample interval (seconds) before resampling. |

## Read it programmatically

<!-- no-validate -->
```python
from pathlib import Path
from dronalize.io import read_manifest

manifest = read_manifest(Path("output"))
print(manifest.trajectory_schema)
print(manifest.trajectory_schema_fields)
print(manifest.storage_backend)
print(manifest.feature_columns)
print(manifest.horizon_frames, manifest.default_observation_length)
```

See [`read_manifest()`](../reference/api/io/storage-and-manifests.md#dronalize.io.read_manifest) for the full reader API.
