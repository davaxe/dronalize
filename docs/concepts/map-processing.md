# Map processing

<div class="section-intro" markdown="1">
Map handling in `dronalize` has two separate concerns: whether a run should include maps at all,
and how much map geometry should be kept around each scene.
</div>

For exact settings, see the [map reference](../reference/configuration/map.md). For dataset support,
see the [dataset reference](../reference/datasets/index.md).

## Inclusion vs. configuration

Map inclusion is a runtime decision:

- if a dataset does not support maps, the runtime ignores map settings
- if a dataset supports maps, `--include-map` and `--no-map` can force inclusion on or off

The `[map]` section does not turn maps on or off. It only configures how maps are extracted when
maps are included.

## Extraction modes

Extraction is configured under `[datasets.<name>.map.extraction]`.

| Mode | When to use it |
| --- | --- |
| `full` | Keep the whole map when size is manageable or you want maximum context. |
| `scene_extent` | Crop adaptively around the scene trajectory extent, with either a circle or bounding box. |
| `circle` | Keep a fixed-radius area around the scene. |
| `trajectory_buffer` | Keep only the road geometry that stays close to the actual scene trajectories. |
| `bounding_box` | Keep a fixed-width, fixed-height area around the scene. |

As a rule of thumb:

- use `scene_extent` for adaptive local context
- use `trajectory_buffer` when you want a smaller cut that follows the driven paths more closely
- use `circle` or `bounding_box` when downstream code expects a consistent spatial extent
- use `full` when cropping is unnecessary or the map is already small

## Geometry density

`min_distance` and `interpolation_distance` control how dense the retained geometry is.

- `min_distance` removes overly dense points
- `interpolation_distance` controls the spacing used when geometry is reconstructed

These settings matter most for storage size, rendering cost, and graph density. If you do not have
a reason to tune them, dataset defaults are usually a good starting point.

## Semantic filtering

The optional `[datasets.<name>.map.edge_types]` block lets you reshape the semantic map payload without touching dataset code.

- use `exclude` to drop noisy edge classes such as `VIRTUAL`
- use `include` when you want a strict semantic subset
- use `remap` to collapse fine-grained types into a smaller vocabulary

## Typical setup

```toml
[datasets.a43.map]
min_distance = 1.0
interpolation_distance = 2.5

[datasets.a43.map.extraction]
mode = "scene_extent"
padding = 1.2
shape = "bounding_box"

[datasets.a43.map.edge_types]
exclude = ["VIRTUAL"]
```

This keeps a scene-aligned bounding box around each scene, simplifies the geometry to the requested
point density, and removes virtual edges from the emitted graph.
