# Configuration model

<div class="section-intro" markdown="1">
Configuration in `dronalize` is layered. Every run starts from dataset defaults, then optional
profiles are applied, then dataset-local TOML settings refine the result, and finally runtime
overrides can change selected fields for one run.
</div>

For exact field tables and TOML syntax, see the [configuration reference](../reference/configuration/index.md).

## Mental model

Think of one resolved dataset config as four layers:

1. built-in dataset defaults from the
   [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor)
2. profile fragments from `[profiles.<name>]`
3. one dataset entry under `[datasets.<name>]`
4. runtime overrides from the CLI or
   [`RuntimeOverride`](../reference/api/config/project.md#dronalize.config.RuntimeOverride)

Profiles are opted into with `uses`, in the order they should be applied.

## Current top-level sections

| Section | Purpose |
| --- | --- |
| `runtime` | Worker count and execution tuning. |
| `scenes` | Scene window shape, sliding-window extraction, resampling, and lane-change sampling. |
| `screening` | Cleanup rules, scene checks, and agent checks. |
| `map` | Map extraction mode and geometry density. |
| `read` | Raw input selection, including dataset-native partitions. |
| `assign` | Train, val, and test output routing. |
| `output` | Output schema, precision, recentering, and MDS-specific tuning. |
| `loader_options` | Dataset-owned loader options validated by the selected dataset integration. |

## Profiles and dataset entries

Use profiles for shared policy:

```toml
[profiles.fast.runtime]
jobs = "auto"
```

Then opt a dataset into that profile:

```toml
[datasets.a43]
uses = ["fast"]
```

Add dataset-local settings where the dataset should differ:

```toml
[datasets.a43.scenes]
history_frames = 20
future_frames = 60
sample_time = 0.1
```

You can combine multiple profiles:

```toml
[profiles.fast.runtime]
jobs = "auto"

[profiles.high_precision.output]
schema = "positions_velocity_yaw"
precision = "float64"

[datasets.a43]
uses = ["fast", "high_precision"]
```

## Example

```toml
[profiles.fast.runtime]
jobs = "auto"

[profiles.high_precision.output]
schema = "positions_velocity_yaw"
precision = "float64"
recenter_positions = true

[datasets.a43]
uses = ["fast", "high_precision"]

[datasets.a43.scenes]
history_frames = 20
future_frames = 60
sample_time = 0.1

[datasets.a43.scenes.window]
step = 2

[datasets.a43.assign]
strategy = "shuffled-time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
segments = 8
gap = 2

[datasets.a43.map.extraction]
mode = "circle"
radius = 60.0
```

## How merging works

The merge behavior depends on the section:

- nested config sections such as `scenes`, `runtime`, `map`, and `output` merge into inherited values
- `read` and `assign` are replaced as whole strategy configs
- `screening` can either replace or extend inherited named rules
- `loader_options` is dataset-owned and validated by the selected dataset integration

That makes it practical to change only the pieces you care about while keeping the dataset's built-in
defaults intact.

## Runtime overrides

The CLI and Python runtime can override a focused subset of values without editing the TOML file:

- read and assignment strategy parameters
- worker count
- output schema
- map inclusion

This is useful for quick experiments, but the stable baseline should still live in the dataset's
default config and your TOML file.

## Practical workflow

1. Start from the dataset defaults and inspect them with `dronalize inspect <dataset>`.
2. Add a minimal `[datasets.<name>]` entry that changes only what your project needs.
3. Move repeated policy into profiles and opt datasets in with `uses`.
4. Use CLI overrides for temporary run-to-run experiments, not for long-term project config.
