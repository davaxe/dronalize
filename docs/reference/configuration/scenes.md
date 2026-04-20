# `[scenes]` section

<div class="section-intro" markdown="1">
The scenes config describes how raw trajectory data becomes model-ready scenes. In practice, it controls four things: scene-window length, optional temporal transforms such as windowing and resampling behavior.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `history_frames` | `int` | Number of history (observed) frames per scene window before resampling. | `dataset default` |
| `future_frames` | `int` | Number of future frames (frames to predict) per scene window before resampling. | `dataset default` |
| `sample_time` | `float` | Time between frames before resampling. | `dataset default` |
| `window` | `table` or `false` | Sliding-window extraction settings. Use `false` to disable an inherited window block. | `inherited` |
| `resample` | `table` or `false` | Temporal resampling and interpolation settings. Use `false` to disable an inherited resample block. | `inherited` |
| `lane_change` | `table` or `false` | Lane-change sampling controls for highway-style datasets. Use `false` to disable an inherited lane-change block. | `inherited` |

The `scenes` block merges into the dataset's built-in scene config, so several effective defaults are dataset-specific rather than global.

For dataset-specific defaults and dataset-owned loader behavior, see the [dataset reference](../datasets/index.md).

## Most common setup

For many datasets, the most important loader settings are only these:

```toml
[datasets.a43.scenes]
history_frames = 20
future_frames = 60
sample_time = 0.1
```

That defines one scene window as:

- `20` observed frames
- `60` future frames
- sampled at `0.1` seconds per frame

## `[scenes.window]` section

Use windowing when a longer recording should generate multiple overlapping scene windows.

| Key | Type | Description | Default |
|---|---|---|---|
| `step` | `int` | Number of frames to advance between consecutive windows. | `required` |

!!! note "Window size"
    Window size is derived automatically as `history_frames + future_frames`. When windowing is used, the window size is fixed and the `step` controls how much the window slides forward each time.

Example:

```toml
[datasets.a43.scenes]
history_frames = 20
future_frames = 60

[datasets.a43.scenes.window]
step = 2
```

This keeps the scene-window length at `80` total frames and slides the window forward by `2` frames each time.

## `[scenes.resample]` section

Use resampling when data should be interpolated to a different temporal resolution before scenes are produced.

| Key | Type | Description | Default |
|---|---|---|---|
| `up` | `int` | Upsampling factor. | `1` |
| `down` | `int` | Downsampling factor. | `1` |
| `method` | `"linear"`, `"cubic"`, or `"pchip"` | Interpolation method to use when interpolating. | `"linear"` |
| `coordinates` | `array[str]` | Position columns used as the resampling base. | `["x", "y"]` |
| `max_gap` | `int` | Maximum frame gap before a new resampling segment is started. | `1` |
| `emit_velocity` | `bool` | Whether to generate velocity columns as first-order output derivatives. Can only be used for spline-based resampling. | `false` |
| `emit_acceleration` | `bool` | Whether to generate acceleration columns as second-order output derivatives. Can only be used for spline-based resampling. | `false` |

!!! "Specifying coordinates"
  The coordinate keys must match the position column names in the source dataset. For example, if the dataset uses `x1` and `x2` for positions instead of `x` and `y`, those keys should be specified here.
  Usually, it is preferred to use the standard `x` and `y` column names for positions, but this option allows resampling to work with datasets that use different naming conventions without requiring a custom dataset spec.

Example:

```toml
[datasets.a43.scenes.resample]
up = 2
down = 1
method = "cubic"
max_gap = 4
coordinates = ["x1", "x2"]
```

!!! note  "When to use `emit_velocity` and `emit_acceleration`"
    These options are only relevant for spline-based resampling methods such as `"cubic"` and
    `"pchip"`. They add velocity and acceleration columns derived analytically from the interpolation
    fit. They are invalid for `"linear"` resampling.
    
    [`scipy.interpolate`](https://docs.scipy.org/doc/scipy/reference/interpolate.html) is used for spline
    resampling.
    
## `[scenes.lane_change]` section

Use lane-change sampling only for lane-change-oriented highway datasets that expose this behavior.

If a dataset does not support lane-change sampling, adding this block is a configuration error.

| Key | Type | Description | Default |
|---|---|---|---|
| `persist` | `int` | Number of frames a lane change must persist to count as a positive event. | `required` |
| `margin_before` | `int` | Required number of frames before the lane change event. | `0` |
| `margin_after` | `int` | Required number of frames after the lane change event. | `0` |
| `required_lane_changes` | `int` | Minimum number of lane change events required for a positive scene window. | `1` |
| `negative_keep_every` | `int` | Keep every Nth negative scene window. Set to `1` to keep all negatives. | `3` |

## Example

```toml
[datasets.a43.scenes]
history_frames = 20
future_frames = 60
sample_time = 0.1

[datasets.a43.scenes.window]
step = 2

[datasets.a43.scenes.resample]
up = 2
down = 1
method = "cubic"
```

To disable an inherited optional block instead of overriding it, set the key to `false` on the
parent `[...scenes]` table:

```toml
[datasets.highd.scenes]
resample = false
lane_change = false
```
