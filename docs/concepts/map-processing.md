# Map processing

<div class="section-intro" markdown="1">
Map processing is optional. When a dataset supports maps, the main decisions are whether to include map context at all, how much geometry to keep, and how tightly to crop the map around each scene.
</div>

For exact settings and extraction fields, see the [map reference](../reference/configuration/map.md). For dataset support, see the [dataset reference](../reference/datasets/index.md).

## Mental model

Map configuration answers three questions:

1. Should maps be included?
2. How dense or simplified should the geometry be?
3. How much of the map should be kept around each scene?

Most of the time, the most important choice is the extraction mode.

## Choosing an extraction mode

| Mode | When to use it |
| --- | --- |
| `full` | Keep the entire map when map size is manageable or you want maximum context. |
| `relevant` | Keep only the map around the scene itself. Good when you want local context without choosing a fixed crop size. |
| `circle` | Keep a fixed-radius area around the scene. Good when your model expects a consistent spatial extent. |
| `bounding_box` | Keep a fixed-width, fixed-height area. Good when a rectangular crop fits downstream assumptions better than a circle. |

As a rule of thumb:

- choose `relevant` for adaptive local context
- choose `circle` or `bounding_box` for fixed-size context
- choose `full` when cropping is unnecessary

## Geometry controls

`min_distance` and `interp_distance` control how detailed the retained map geometry is.

- `min_distance` removes overly dense points
- `interp_distance` controls how geometry is reconstructed between retained points

These settings matter most when map size, smoothness, or storage cost becomes important. Otherwise, dataset defaults are often a good starting point.

## Typical setup

```toml
[datasets.a43.map]
enabled = true
extraction = "circle"
radius = 60.0
```

This is a common pattern when you want map context, but only within a fixed area around each scene.
