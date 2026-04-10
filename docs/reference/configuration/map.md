# `[map]` section

<div class="section-intro" markdown="1">
Map settings control whether map data is included and, if it is, how much of the map is kept around each scene. The most important choice here is usually the extraction mode.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `min_distance` | `float` | Minimum spacing used when simplifying map geometry. Must be greater than `0`. | `inherited` |
| `interp_distance` | `float` | Interpolation spacing used when densifying geometry. Must be greater than or equal to `min_distance`. | `inherited` |

!!! note "Disable map data"
    Currently teh map data cannot be disabled trough the config. To exclude map data use `--no-map` when
    using the CLI.

## [`map.extraction`] section

The extraction decides whether the full map is kept or if cropped around the current scene. It is defined
by a `mode` and a set of parameters that depend on the mode.

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `str` | Map extraction mode. | `"full"` |

The available modes are:

- `"full"` keeps the full map without cropping. This have no additional parameters.
- `"scene_extent"` crops around the scene trajectory extent and requires `padding`.
- `"circle"` crops with a circular area and requires `radius`.
- `"bounding_box"` crops with an axis-aligned box and requires both `width` and `height`.

### `mode` = `"scene_extent"`

When `mode` is set to `"scene_extent"`, the map is cropped around the bounding box of the scene trajectory, with an additional `padding` distance added in all directions.

| Key | Type | Description | Default |
|---|---|---|---|
| `padding` | `float` | Factor to inflate the scene trajectory bounding box when cropping the map. Must be greater than `1.0`. | `1.15` |

### `mode` = `"circle"`

| Key | Type | Description | Default |
|---|---|---|---|
| `radius` | `float` | Radius of the circular area to crop around the scene trajectory. Must be greater than `0`. | `required` |

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
mode = "circle"
radius = 50.0
```
