# `[map]` section

<div class="section-intro" markdown="1">
Map settings control how map data is processed when a run includes maps. The most important choice here is usually the extraction mode.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `min_distance` | `float` | Minimum spacing used when simplifying map geometry. Must be greater than `0`. | `inherited` |
| `interp_distance` | `float` | Interpolation spacing used when densifying geometry. Must be greater than or equal to `min_distance`. | `inherited` |
| `edge_types` | `table` | Optional semantic edge filtering and remapping rules. | `none` |

!!! note "Disable map data"
    Map data cannot currently be disabled through TOML. To exclude maps at runtime, use `--no-map`
    on the CLI or pass `include_map=False` in the Python runtime request.

## [`map.extraction`] section

The extraction decides whether the full map is kept or if cropped around the current scene. It is defined
by a `mode` and a set of parameters that depend on the mode.

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `str` | Map extraction mode. | `"full"` |

The available modes are:

- `"full"` keeps the full map without cropping. This has no additional parameters.
- `"scene_extent"` crops around the scene trajectory extent and requires `padding`.
- `"circle"` crops with a circular area and requires `radius`.
- `"trajectory_buffer"` keeps only map nodes within a fixed radius of the scene trajectories.
- `"bounding_box"` crops with an axis-aligned box and requires both `width` and `height`.

### `mode` = `"scene_extent"`

When `mode` is set to `"scene_extent"`, the map is cropped adaptively around the scene trajectory. The adaptive crop can be a circle around the scene centroid or a bounding box around the scene extent.

| Key | Type | Description | Default |
|---|---|---|---|
| `padding` | `float` | Factor applied to the adaptive crop extent. Must be greater than or equal to `1.0`. | `1.0` |
| `shape` | `"circle"` or `"bounding_box"` | Shape used for the adaptive crop. | `"circle"` |

### `mode` = `"circle"`

| Key | Type | Description | Default |
|---|---|---|---|
| `radius` | `float` | Radius of the circular area to crop around the scene trajectory. Must be greater than `0`. | `required` |

### `mode` = `"trajectory_buffer"`

| Key | Type | Description | Default |
|---|---|---|---|
| `radius` | `float` | Buffer radius around each relevant trajectory point. Must be greater than `0`. | `required` |

### `mode` = `"bounding_box"`

| Key | Type | Description | Default |
|---|---|---|---|
| `width` | `float` | Width of the axis-aligned bounding box to crop around the scene trajectory. Must be greater than `0`. | `required` |
| `height` | `float` | Height of the axis-aligned bounding box to crop around the scene trajectory. Must be greater than `0`. | `required` |

## Minimal example

```toml
[datasets.a43.map]
min_distance = 1.0
interp_distance = 2.5

[datasets.a43.map.extraction]
mode = "trajectory_buffer"
radius = 8.0
```

## [`map.edge_types`] section

Use this section when you want to simplify the semantic map vocabulary without changing the dataset loader itself.

| Key | Type | Description | Default |
|---|---|---|---|
| `include` | `list[str]` | Optional allow-list of edge types to keep after remapping. | `all` |
| `exclude` | `list[str]` | Edge types to drop after remapping. | `[]` |
| `remap` | `table[str, str]` | Mapping from one edge type to another before include/exclude is applied. | `{}` |

Edge types use the shared `EdgeType` names such as `CURB`, `ROAD_BORDER`, `VIRTUAL`, or `LINE_THIN_DOUBLE`.

```toml
[datasets.a43.map.edge_types]
exclude = ["VIRTUAL"]

[datasets.a43.map.edge_types.remap]
LINE_THIN_DOUBLE = "LINE_THIN"
LINE_THIN_DOUBLE_DASHED = "LINE_THIN_DASHED"
```
