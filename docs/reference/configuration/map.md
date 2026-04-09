# `[map]` section

<div class="section-intro" markdown="1">
Map settings control whether map data is included and, if it is, how much of the map is kept around each scene. The most important choice here is usually the extraction mode.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `enabled` | `bool` | Enable or disable map processing for the dataset. Set to `false` to disable maps entirely. | `inherited` |
| `min_distance` | `float` | Minimum spacing used when simplifying map geometry. Must be greater than `0`. | `inherited` |
| `interp_distance` | `float` | Interpolation spacing used when densifying geometry. Must be greater than or equal to `min_distance`. | `inherited` |
| `extraction` | `"full"`, `"scene_extent"`, `"circle"` or `"bounding_box"` | Map extraction mode. | `inherited` |
| `padding` | `float` | Padding factor around the scene trajectory extent when `extraction = "scene_extent"`. | `required` |
| `radius` | `float` | Radius in map units when `extraction = "circle"`. | `required` |
| `width` | `float` | Extraction width when `extraction = "bounding_box"`. | `required` |
| `height` | `float` | Extraction height when `extraction = "bounding_box"`. | `required` |

The `map` section is merged into the dataset's default map configuration. If a dataset does not already carry a map config and you enable it from file config, Dronalize creates a default full-map configuration first.

## Extraction modes

The extraction mode decides whether Dronalize keeps the full map or crops it around the current scene.

- `"full"` keeps the full map without cropping.
- `"scene_extent"` crops around the scene trajectory extent and requires `padding`.
- `"circle"` crops with a circular area and requires `radius`.
- `"bounding_box"` crops with an axis-aligned box and requires both `width` and `height`.

!!! note "Validation"
    Extraction-specific fields are only valid when the matching `extraction` mode is set explicitly.

## Minimal example

```toml
[datasets.a43.map]
enabled = true
min_distance = 1.0
interp_distance = 2.5
extraction = "circle"
radius = 60.0
```
