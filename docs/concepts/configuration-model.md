# Configuration model

<div class="section-intro" markdown="1">
Configuration in `dronalize` is layered. Every run starts from dataset defaults, then optional shared settings are applied, then dataset-specific overrides refine the final behavior.
</div>

For exact TOML syntax and field tables, see the [configuration reference](../reference/configuration/index.md), [loader](../reference/configuration/loader.md), [map](../reference/configuration/map.md), [split](../reference/configuration/split.md), [export](../reference/configuration/export.md), [execution](../reference/configuration/execution.md), and [filter](../reference/configuration/filter.md) reference pages.

## Mental model

Think of configuration as three layers:

1. Built-in dataset defaults
2. optional `profiles` composed in declared order
3. `[datasets.<name>]` settings for one dataset only

This lets you keep a common baseline while still handling dataset-specific differences cleanly.

## How to organize a config file

Use `[profiles.<name>]` when the same setting should be reusable across datasets.

```toml
[profiles.fast.execution]
jobs = "auto"
```

Then opt a dataset into that profile when needed.

```toml
[datasets.a43]
extends = ["fast"]

[datasets.a43.loader]
history_frames = 20
future_frames = 60
sample_time = 0.1
```

You can combine both:

```toml
[profiles.fast.execution]
jobs = "auto"

[datasets.a43]
extends = ["fast"]

[datasets.a43.loader]
history_frames = 20
future_frames = 60
```

In practice:

- `profiles` are reusable policies
- `datasets.<name>` is a local exception

## What the main sections do

| Section | Purpose |
| --- | --- |
| `execution` | Controls how work is distributed. |
| `loader` | Defines sample shape, temporal transforms, filtering, and dataset-specific loading behavior. |
| `map` | Controls whether map context is included and how much of it is kept. |
| `split` | Routes data into train, val, and test outputs. |
| `export` | Controls the saved schema, precision, and storage settings. |

## Practical workflow

For most projects, the easiest approach is:

1. Start with one dataset and no configuration or a minimal `loader` block.
2. Add `split` and `export` once you know how you want to train and save data. 
3. Add `map` only if the dataset supports map context and you need it.
4. Move repeated settings into `profiles` when multiple datasets should share them.

Many nested sections merge into inherited defaults rather than forcing you to redefine everything. That makes it practical to override only the parts you want to change.

!!! note "CLI overrides"
    The CLI also support overriding a specific subset of config values without editing the
    file. This is useful for quick experiments or when you want to keep a single
    config file but run with different settings.

## Multi-dataset configuration example

This example demonstrates how to configure settings for multiple datasets in the
same file. It first sets a shared execution policy, then defines specific loader
settings for two datasets, `a43` and `argoverse1`. The `argoverse1` dataset also
includes dataset-specific `dataset` config and a resampling block to match the
sample time of `a43`.

The `a43` dataset overrides the shared execution policy by setting `jobs` to 1, which means it will not use parallel processing.
This can be useful if a dataset is known to be small and parallel processing would not provide significant speedup.

```toml
[profiles.fast.execution]
jobs = "auto"

[datasets.a43]
extends = ["fast"]

[datasets.a43.execution]
jobs = 1

[datasets.a43.loader]
history_frames = 20
future_frames = 60
sample_time = 0.1

[datasets.argoverse1.loader]
history_frames = 6
future_frames = 10
sample_time = 0.5

[datasets.argoverse1.dataset]
file_batch_size = 8

[datasets.argoverse1.loader.resampling]
method = "cubic"
up = 5
down = 1
```
