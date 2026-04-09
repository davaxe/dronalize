# Configuration Reference

<div class="section-intro" markdown="1">
This reference describes the TOML configuration surface used by `dronalize`. The file is validated before processing starts, with structured schema checks handling field types, nested table shapes, and section-specific requirements so you can catch configuration issues early.

Dataset-specific defaults, capabilities, and dataset-owned configuration behavior are documented in the [dataset reference](../datasets/index.md).
</div>

## Expected file shape

Authoring config has two top-level entry points:

- `global` for settings that apply to every dataset
- `datasets.<dataset-name>` for settings that apply only to one dataset key

Minimal shape:

```toml
[global]

[datasets.<dataset-name>]
```

Most useful config lives in nested blocks under one of those roots:

```toml
[global.execution]
jobs = 4

[datasets.a43.loader]
input_len = 20
output_len = 60
sample_time = 0.1

[datasets.a43.export]
schema = "positions_velocity_yaw"
precision = "float64"
```

In other words:

- `[global.execution]` affects every dataset unless a dataset-specific block overrides it.
- `[datasets.a43.loader]` affects only the `a43` dataset.
- Nested tables continue from there, for example `[datasets.a43.loader.filter]` or `[datasets.a43.export.mds]`.


## Available section roots

Inside either `global` or `datasets.<dataset-name>`, the current section roots are:

| Block | Purpose |
| --- | --- |
| `execution` | Worker count and executor chunking. |
| `loader` | Horizons, sample time, windowing, resampling, filtering, lane-change sampling, and dataset-specific loader options. |
| `map` | Map enablement and extraction settings. |
| `export` | Persisted trajectory schema, precision, offsets, and MDS backend tuning. |
| `split` | Split strategy, ratios, native split selection, and temporal split parameters. |

!!! note "Section roots"
    In general, roots can be left unspecfied (not present in the file), but if
    added they must be filled with valid keys and values for all of their
    required fields.

## How overrides are applied

The reference pages describe the file format only, but the values are resolved in layers:

1. Dataset spec defaults
2. `[global]`
3. `[datasets.<dataset-name>]`

That matters because many fields do not start from a single hard-coded default. Instead, they begin from the dataset's built-in runtime config and are then overridden or merged by the configuration file.

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

- `[loader]` means the table directly under `[global]` or `[datasets.<dataset-name>]`
- `[loader.resampling]` means a nested table inside `loader`
- `[[loader.filter.agent]]` means an array-of-tables entry, so you can define that block more than once

For example:

```toml
[datasets.a43.loader.filter]
mode = "extend"

[[datasets.a43.loader.filter.agent]]
type = "min_samples"
minimum = 8
rule_id = "sample_floor"

[[datasets.a43.loader.filter.agent]]
type = "window"
start_frame = 0
end_frame = 19
min_fraction = 0.8
```

This creates one `loader.filter` table with two separate `agent` rule entries.

## Example

```toml
[global.execution]
jobs = "auto"

[datasets.a43.loader]
input_len = 20
output_len = 60
sample_time = 0.1

[datasets.a43.map]
enabled = true
extraction = "circle"
radius = 60.0

[datasets.a43.split]
strategy = "shuffled-time"
ratio = { train = 0.7, val = 0.2, test = 0.1 }
segments = 8

[datasets.a43.export]
schema = "positions_velocity_yaw"
precision = "float64"
```
