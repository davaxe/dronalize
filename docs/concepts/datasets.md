# Datasets

<div class="section-intro" markdown="1">
A dataset key in `dronalize` is not just a name. It resolves to a
[`DatasetSpec`](../reference/api/datasets/spec.md#dronalize.datasets.DatasetSpec), which defines how that dataset
is discovered, loaded, configured, and processed.
</div>

For the full built-in dataset list and dataset-specific notes, see the
[dataset reference](../reference/datasets/index.md).

## What a [`DatasetSpec`](../reference/api/datasets/spec.md#dronalize.datasets.DatasetSpec) provides

Each [`DatasetSpec`](../reference/api/datasets/spec.md#dronalize.datasets.DatasetSpec) carries the dataset
integration contract:

- `default_config` for the dataset's starting point
- `native_schema` for the loader's physical trajectory fields
- `supported_native_splits` when the dataset ships with fixed partitions
- `feature_support` for optional capabilities such as map data and lane-change sampling
- `dataset_options_model` for typed `[datasets.<name>.dataset]` config
- `resources_factory` for run-scoped shared resources such as maps
- `split_support` when scene-, source-, or time-based split modes are valid

This is why configuration often starts small: the dataset already provides a meaningful default
window, schema, screening policy, and map setup.

## Registry behavior

The registry is lazy. Built-in dataset modules are imported only when a dataset is first requested.
That keeps optional dependencies isolated.

In practice:

- [`dronalize.datasets.get("waymo")`](../reference/api/datasets/registry.md#dronalize.datasets.get) requires the
  `waymo` extra
- [`dronalize.datasets.available()`](../reference/api/datasets/registry.md#dronalize.datasets.available) only lists
  built-ins whose optional dependencies are installed
- the CLI commands `available`, `inspect`, and `split-support` are direct views of the registry

## Why dataset choice matters

Different datasets support different workflows. Common differences are:

- native benchmark splits vs. source discovery only
- map support vs. no map support
- time-based split support vs. only scene or source routing
- dataset-owned config options
- lane-change-aware defaults for some highway datasets

So choosing a dataset key also chooses the capabilities you can rely on.

## Practical workflow

When starting with a dataset:

1. Run `dronalize inspect <dataset>` to see its defaults and capabilities.
2. Run `dronalize split-support <dataset>` before choosing a split strategy.
3. Keep your TOML focused on project-specific changes instead of repeating built-in defaults.
4. Use `show-config` or `resolve_request()` to verify the fully merged result before execution.
