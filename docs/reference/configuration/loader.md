# `[loader]` section

<div class="section-intro" markdown="1">
The loader config describes how raw trajectory data becomes model-ready scenes. In practice, it controls four things: scene-window length, optional temporal transforms such as windowing and resampling, optional filtering, and dataset-specific loader behavior.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `input_len` | `int` | Number of observed frames per scene window before resampling. | `dataset default` |
| `output_len` | `int` | Number of predicted frames per scene window before resampling. | `dataset default` |
| `sample_time` | `float` | Time between frames before resampling. | `dataset default` |
| `window` | `table` | Sliding-window extraction settings. | `inherited` |
| `resampling` | `table` | Temporal resampling and interpolation settings. | `inherited` |
| `filter` | `table` | Cleanup and validation rules applied during loading. | `inherited` |
| `lane_change_sampling` | `table` | Lane-change sampling controls for highway-style datasets. | `inherited` |
| `options` | `table` | Dataset-dependent loader options validated against the selected dataset's option schema. | `none` |

The `loader` block merges into the dataset's built-in loader config, so several effective defaults are dataset-specific rather than global.

For dataset-specific defaults and dataset-owned loader behavior, see the [dataset reference](../datasets/index.md).

## Most common setup

For many datasets, the most important loader settings are only these:

```toml
[datasets.a43.loader]
input_len = 20
output_len = 60
sample_time = 0.1
```

That defines one scene window as:

- `20` observed frames
- `60` future frames
- sampled at `0.1` seconds per frame

## `[loader.window]` section

Use windowing when a longer recording should generate multiple overlapping scene windows.

| Key | Type | Description | Default |
|---|---|---|---|
| `size` | `int` | Total number of frames in each extracted window. | `required` |
| `step` | `int` | Number of frames to advance between consecutive windows. | `required` |

!!! note "Window size"
    `size` must equal `input_len + output_len`.

Example:

```toml
[datasets.a43.loader]
input_len = 20
output_len = 60

[datasets.a43.loader.window]
size = 80
step = 2
```

This keeps the scene-window length at `80` total frames and slides the window forward by `2` frames each time.

## `[loader.resampling]` section

Use resampling when data should be interpolated to a different temporal resolution before scenes are produced.

| Key | Type | Description | Default |
|---|---|---|---|
| `up` | `int` | Upsampling factor. | `1` |
| `down` | `int` | Downsampling factor. | `1` |
| `method` | `"linear"`, `"cubic"`, `"pchip"` or `"hermite"` | Interpolation method. Some aliases are also accepted. | `"linear"` |
| `position_columns` | `array[str]` | Position columns used as the resampling base. | `["x", "y"]` |
| `input_derivatives` | `array[table]` | Input derivative columns, defined with array-of-table entries. | `none` |
| `output_derivatives` | `array[table]` | Output derivative columns to generate, defined with array-of-table entries. | `none` |
| `max_gap` | `int` | Maximum frame gap before a new resampling segment is started. | `1` |
| `sort` | `bool` | Sort data by frame before segmenting for resampling. | `true` |
| `sample_time` | `float` | Sample time stored on the resampling spec. | `1.0` |

### Choosing a method

- `"linear"` is the simplest option and does not support derivative inputs or outputs.
- `"cubic"` and `"pchip"` support generated `output_derivatives`.
- `"hermite"` requires exactly first-order `input_derivatives`.

Accepted aliases include `fast` for linear, `spline` and `cubic_spline` for cubic, `pchip_interpolation` for PCHIP, and `hermite_spline` for Hermite.

### `[[loader.resampling.input_derivatives]]` and `[[loader.resampling.output_derivatives]]`

| Key | Type | Description | Default |
|---|---|---|---|
| `order` | `int` | Positive derivative order. | `required` |
| `columns` | `array[str]` | Output column names for that derivative order. | `required` |

Example:

```toml
[datasets.a43.loader.resampling]
up = 2
down = 1
method = "cubic"
max_gap = 4
position_columns = ["x", "y"]

[[datasets.a43.loader.resampling.output_derivatives]]
order = 1
columns = ["vx", "vy"]

[[datasets.a43.loader.resampling.output_derivatives]]
order = 2
columns = ["ax", "ay"]
```

!!! note  "When to use input- and output derivatives"
    `input_derivative` can only be used when the source dataset already include
    derivative columns and the selected method supports them, i.e., `"hermite"`. `output_derivative` can be used with any spline based
    method to directly generate analytical derivatives based on the spline definition.
    
    [`scipy.interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) is used for spline
    resampling.
    
## `[loader.filter]` section

Filtering is part of the loader because it decides what data survives preprocessing before scenes are written.

| Key | Type | Description | Default |
|---|---|---|---|
| `mode` | `"replace"` or `"extend"` | Whether new rules replace the existing filter or merge into it. | `required when rules are present` |
| `remove` | `array[str]` | Rule names to remove after merging. | `none` |
| `cleanup` | `array[table]` | Rules that remove rows before validation. | `none` |
| `scene` | `array[table]` | Scene-level validation rules. | `none` |
| `agent` | `array[table]` | Agent-level validation rules. | `none` |

The full filter rule catalog is documented on the dedicated [filter reference](filter.md) page.

## `[loader.lane_change_sampling]` section

Use `lane_change_sampling` only for lane-change-oriented highway datasets that expose this behavior.

If a dataset does not support lane-change sampling, adding this block is a configuration error.

| Key | Type | Description | Default |
|---|---|---|---|
| `persist` | `int` | Number of frames a lane change must persist to count as a positive event. | `required` |
| `margin_before` | `int` | Required number of frames before the lane change event. | `0` |
| `margin_after` | `int` | Required number of frames after the lane change event. | `0` |
| `required_lane_changes` | `int` | Minimum number of lane change events required for a positive scene window. | `1` |
| `negative_keep_every` | `int` | Keep every Nth negative scene window. Set to `1` to keep all negatives. | `3` |

## `[loader.options]` section

`options` is a dataset-specific table for loader-owned parameters. It is validated against the selected dataset's loader option model and is only valid for datasets that expose loader options.

Loader options remain dataset-specific, so an `options` block that is valid for one dataset may be invalid for another. If a dataset does not expose loader-specific options, adding `[loader.options]` is a configuration error.

Example for a dataset that supports loader options:

```toml
[datasets.argoverse1.loader.options]
file_batch_size = 8
```

Use `dronalize inspect <dataset>` to see whether a dataset exposes loader options and which option keys it supports by default. The [dataset reference](../datasets/index.md) is the place to document dataset-owned loader behavior beyond the shared config surface.

## Example

```toml
[datasets.a43.loader]
input_len = 20
output_len = 60
sample_time = 0.1

[datasets.a43.loader.window]
size = 80
step = 2

[datasets.a43.loader.resampling]
up = 2
down = 1
method = "cubic"
```
