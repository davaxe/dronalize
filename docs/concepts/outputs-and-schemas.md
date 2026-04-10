# Exports, Trajectory Schemas, and Storage Backends

<div class="section-intro" markdown="1">
Export decisions in `dronalize` are easiest to understand as two separate choices: what each saved scene contains, and how those scenes are stored on disk.
</div>

For exact export fields and MDS-specific options, see the [export reference](../reference/configuration/export.md) for the configuration. For CLI usage, including `--storage-backend`, see [First run (CLI)](../start/first-run-cli.md).

## Mental model

There are two independent layers here:

1. The **trajectory schema**, which decides what fields each scene contains
2. The **storage backend**, which decides how those scenes are written to disk

That separation is important. In principle, the same schema could be written through different storage backends.

## Trajectory schema

The schema is the logical shape of each scene.

It answers questions like:

- does the scene include only position, or also velocity, acceleration, or yaw?
- what features will downstream code be able to read directly?
- how small or rich should each saved scene be?

Built-in schemas cover a few common use cases:

- position-only outputs for the smallest useful dataset
- velocity and acceleration variants for kinematic models
- yaw variants when orientation matters
- `canonical` when you want the broadest built-in feature set

Choose the smallest schema that still matches your downstream task. Smaller schemas are usually easier to store, load, and work with. The table below summarizes the built-in schemas and their fields.

| Schema name | Feature fields | Feature dim |
|---|---|---|
| `positions_only` | `x`,`y` | 2 |
| `positions_yaw` | `x`,`y`,`yaw` | 3 |
| `positions_velocity` | `x`,`y`,`vx`,`vy` | 4 |
| `positions_velocity_yaw` | `x`,`y`,`vx`,`vy`,`yaw` | 5 |
| `positions_velocity_acceleration` | `x`,`y`,`vx`,`vy`,`ax`,`ay` | 6 |
| `canonical` | `x`,`y`,`vx`,`vy`,`ax`,`ay`,`yaw` | 7 |

!!! note "Custom schemas"
    If the built-in schemas are not a good fit, you can define a custom schema. The export reference documents the exact syntax and required base fields.
    
## Storage backend

The storage backend is the physical persistence layer.

It answers questions like:

- what files are written?
- how are scenes grouped on disk?
- which backend-specific options are available?

### Available backends

The currently available backends are:

- `mds`: the main persisted backend, see [MDS docs](https://docs.mosaicml.com/projects/streaming/en/stable/index.html) for details,
- `null`: a debug-only backend that doesn't actually write anything.

## What this means in practice

The schema and storage backend should be thought about separately:

- choose the schema based on model and feature needs
- choose the storage backend based on storage and data-loading needs

Today, backend choice is simple because `mds` is the only real persisted backend. That means most practical export decisions still happen in the schema and export settings, not in backend selection.

## Precision and position offsets

`precision` and `recenter_positions` sit alongside schema choice because they affect how values are written, regardless of the storage backend.

`float32` is the practical default for most runs because it keeps output smaller. `float64` is useful when preserving numeric precision matters more than storage size.

`recenter_positions` recenters positions before writing while preserving the offset in the saved output. This is especially helpful when source coordinates are large and downstream models work better with local coordinates.

!!! warning "Low precision without offsets"
    If you use `float32` precision without `recenter_positions`, be aware that large coordinates may lose precision when saved. This can lead to degraded model performance, especially for tasks that require fine-grained spatial understanding. This is particularly the case for datasets that uses [UTM](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system) coordinate system or similar, where coordinates can be in the millions. In those cases, enabling `recenter_positions` is highly recommended to maintain data quality while still benefiting from reduced storage size.

## Typical setup

```toml
[datasets.a43.export]
schema = "positions_velocity_yaw"
precision = "float64"
recenter_positions = true

[datasets.a43.export.backends.mds]
compression = "zstd:7"
```

Conceptually, this means:

- write scenes with position, velocity, and yaw
- store values in double precision
- write them using the current MDS backend
