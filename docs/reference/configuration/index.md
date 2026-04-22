# Configuration reference

<div class="section-intro" markdown="1">
This reference describes the TOML configuration surface used by `dronalize`. The file is validated before processing starts, with structured schema checks handling field types, nested table shapes, and section-specific requirements so you can catch configuration issues early.

Dataset-specific defaults, capabilities, and dataset-owned configuration behavior are documented in the [dataset reference](../datasets/index.md).
</div>

## Expected file shape

Authoring config has two top-level entry points:

- `profiles.<name>` for reusable config fragments
- `datasets.<dataset-name>` for settings that apply only to one dataset key

Minimal shape:

<!-- no-validate -->
```toml
[profiles.<profile-name>]

[datasets.<dataset-name>]
```

Most useful config lives in nested blocks under one of those roots:

```toml
[profiles.fast.runtime]
jobs = 4

[datasets.a43]
uses = ["fast"]

[datasets.a43.scenes]
history_frames = 20
future_frames = 60
sample_time = 0.1

[datasets.a43.output]
schema = "positions_velocity_yaw"
precision = "float64"
```

In other words:

- `[profiles.fast.runtime]` defines reusable execution settings.
- `[datasets.a43]` can opt into one or more profiles with `uses = [...]`.
- `[datasets.a43.scenes]` affects only the `a43` dataset.

## Available section roots

Inside either `profiles.<profile-name>` or `datasets.<dataset-name>`, the current section roots are:

| Block | Purpose |
| --- | --- |
| [`runtime`](./runtime.md) | Worker count and executor chunking. |
| [`scenes`](./scenes.md) | Scene window length, sampling time, and related settings. |
| [`screening`](./screening.md) | Scene + agent screening and cleanup. |
| [`read`](./read.md) | Raw input selection, including dataset-native partition reads. |
| [`assign`](./assign.md) | Output split assignment, ratios, and temporal assignment parameters. |
| [`map`](./map.md) | Map extraction and interpolation settings. |
| [`output`](./output.md) | Persisted trajectory schema, precision, offsets, storage backend tuning. |
| [`dataset`](./dataset.md) | Dataset-specific options that don't fit into the other categories. |

!!! note "Section roots"
    In general, roots can be left unspecified (not present in the file), but if
    added they must be filled with valid keys and values for all of their
    required fields.

## How overrides are applied

The reference pages describe the file format only, but the values are resolved in layers:

1. Dataset spec defaults
2. inherited profiles in declared order
3. `[datasets.<dataset-name>]`
4. runtime overrides (very limited set of fields that can be overridden at runtime)

That matters because many fields do not start from a single hard-coded default. Instead, they begin from the dataset's built-in runtime config and are then overridden or merged by the configuration file.

For optional inherited blocks such as `screening`, `scenes.window`, `scenes.resample`, and
`scenes.lane_change`, you can explicitly disable the inherited block with `false` on the parent
table instead of inheriting it:

```toml
[datasets.highd]
screening = false

[datasets.highd.scenes]
resample = false
lane_change = false
```

When you need to know what a specific dataset starts with, or whether it exposes dataset-specific behavior beyond the generic config tables here, check the [dataset reference](../datasets/index.md).

## How to read the section tables

Each section page uses the same columns:

| Column | Meaning |
| --- | --- |
| `Key` | TOML key accepted in that table. |
| `Type` | Accepted TOML shape or value family. |
| `Description` | What the key controls. |
| `Default` | What happens when you omit the key. |

The `Default` column uses a few different notations:

| Default notation | Meaning |
| --- | --- |
| A literal value such as `"float32"`, `0`, or `true` | The code supplies that concrete default directly. |
| `dataset default` | The value comes from the dataset spec's built-in loader config. Different datasets may start with different values. |
| `inherited` | The value comes from the already-resolved parent runtime config. This is common for nested blocks that merge into existing defaults. |
| `required` | You must provide the key when that table or mode is used. |
| `required for ...` | The key is only mandatory in specific modes or shapes. |
| `none` | No value is configured unless you add one. |

Some pages also describe validation rules in notes below the table. Those notes are important because a key may be valid only together with a specific strategy or sibling field.

## Common nesting notation

The page titles use TOML path notation:

- `[scenes]` means the table directly under a dataset entry such as `[datasets.<dataset-name>.scenes]` or `[profiles.<profile-name>.scenes]`
- `[scenes.resample]` means a nested table inside `scenes` section, i.e., `[datasets.<dataset-name>.scenes.resample]`

## Example

```toml
[profiles.global.runtime]
jobs = "auto"

[datasets.a43]
uses = ["global"]

[datasets.a43.scenes]
history_frames = 20
future_frames = 60
sample_time = 0.1

[datasets.a43.map.extraction]
mode = "circle"
radius = 60.0

[datasets.a43.assign]
strategy = "shuffled-time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
segments = 8

[datasets.a43.output]
schema = "positions_velocity_yaw"
precision = "float64"
```
