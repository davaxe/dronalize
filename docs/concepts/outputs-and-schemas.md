# Outputs, schemas, and formats

<div class="section-intro" markdown="1">
Output decisions in `dronalize` are easiest to understand as two separate choices: what each saved sample contains, and how those samples are stored on disk.
</div>

For exact writer fields and MDS-specific options, see the [writer reference](../reference/configuration/writer.md). For CLI usage, including `--output-format`, see [First run (CLI)](../start/first-run-cli.md).

## Mental model

There are two independent layers here:

1. The **scene schema**, which decides what fields each sample contains
2. The **output format**, which decides how those samples are written to disk

That separation is important. In principle, the same schema could be written through different storage backends.

## Scene schema

The schema is the logical shape of each scene sample.

It answers questions like:

- does the sample include only position, or also velocity, acceleration, or yaw?
- what features will downstream code be able to read directly?
- how small or rich should each saved sample be?

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
    If the built-in schemas are not a good fit, you can define a custom schema. The writer reference documents the exact syntax and required base fields.
    
## Output format

The output format is the physical storage backend.

It answers questions like:

- what files are written?
- how are samples grouped on disk?
- which backend-specific options are available?

### Available formats

The currently available formats are:

- `mds`: the main go-to format, see [MDS docs](https://docs.mosaicml.com/projects/streaming/en/stable/index.html) for details,
- `dummy`: a debug only format that doesn't actually write anything.

## What this means in practice

The schema and format should be thought about separately:

- choose the schema based on model and feature needs
- choose the format based on storage and data-loading needs

Today, format choice is simple because `mds` is the only real persisted backend. That means most practical output decisions still happen in the schema and writer settings, not in backend selection.

## Precision and position offsets

`precision` and `offset_positions` sit alongside schema choice because they affect how values are written, regardless of the storage backend.

`float32` is the practical default for most runs because it keeps output smaller. `float64` is useful when preserving numeric precision matters more than storage size.

`offset_positions` recenters positions before writing while preserving the offset in the saved output. This is especially helpful when source coordinates are large and downstream models work better with local coordinates.

!!! warning "Low precision without offsets"
    If you use `float32` precision without `offset_positions`, be aware that large coordinates may lose precision when saved. This can lead to degraded model performance, especially for tasks that require fine-grained spatial understanding. This is particularly the case for datasets that uses [UTM](https://en.wikipedia.org/wiki/Universal_Transverse_Mercator_coordinate_system) coordinate system or similar, where coordinates can be in the millions. In those cases, enabling `offset_positions` is highly recommended to maintain data quality while still benefiting from reduced storage size.

## Typical setup

```toml
[datasets.a43.writer]
schema = "positions_velocity_yaw"
precision = "float64"
offset_positions = true

[datasets.a43.writer.mds]
compression = "zstd:7"
```

Conceptually, this means:

- write scenes with position, velocity, and yaw
- store values in double precision
- write them using the current MDS backend
