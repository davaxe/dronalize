# `[output]` section

<div class="section-intro" markdown="1">
Output settings control what is persisted after preprocessing. In practice, this means choosing the trajectory schema, numeric precision, and any backend-specific output options.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `schema` | `str` or `table` | Trajectory schema to persist. Use either a predefined schema name or a structured custom schema definition. | `"canonical"` |
| `precision` | `"float32"` or `"float64"` | Floating-point precision of persisted data. | `"float32"` |
| `recenter_positions` | `bool` | Offset all agent positions by the scene mean before writing. The offset is stored in the output. | `true` |

## `[export.backends.mds]` section

This nested block only matters when writing MDS output and is used to tune shard writing behavior.

| Key | Type | Description | Default |
|---|---|---|---|
| `compression` | `str` | Compression setting for MDS shards, for example `"zstd:7"`. | `none` |
| `hashes` | `array[str]` | Hash algorithms to apply to MDS shard files. | `none` |
| `size_limit` | `int` or `str` | Shard size limit, in bytes or as the backend-supported size string. | `67_108_864` |
| `exist_ok` | `bool` | Overwrite existing shard files when set to `true`. | `false` |

!!! note "MDS parameters"
    This section can be used to override MDS-specific parameters for the MDS
    storage backend. These values are only used when the selected storage
    backend is `mds`.

## Minimal example

```toml
[datasets.a43.export]
schema = "positions_velocity_yaw"
precision = "float64"
recenter_positions = true

[datasets.a43.export.backends.mds]
compression = "zstd:7"
hashes = ["sha1", "xxh64"]
size_limit = 33554432
exist_ok = true
```

To use a custom schema instead of a predefined one it is possible to define
a custom schema using a table. Custom schemas must include the base fields
`frame`, `id`, `agent_category`, `x`, and `y`:

```toml
[datasets.a43.export]
schema = { name = "custom", fields = ["frame", "id", "agent_category", "x", "y", "vx", "vy"] }
precision = "float64"
recenter_positions = true
```
