# Datasets

<div class="section-intro" markdown="1">
A dataset key in `dronalize` is more than a name. It selects a built-in dataset definition with defaults, capabilities, expected input structure, and supported processing behaviors.
</div>

For the full dataset list and dataset-specific details, see the [dataset reference](../reference/datasets/index.md).

## What a dataset definition gives you

Each built-in dataset comes with a starting point for processing, including:

- default loader behavior
- optional map support
- supported split strategies
- a native trajectory schema
- any dataset-specific loader behavior

This is why configuration often starts small: the dataset already provides sensible defaults.

## Why dataset choice matters

Different datasets support different workflows.

Common differences include:

- some datasets include map context and some do not
- some expose native train, val, and test splits
- some work best with time-based splitting, while others support scene-based or source-based splitting
- some expose dataset-specific loader options or highway-style sampling behavior

In other words, choosing a dataset key also chooses the set of features you can reasonably use.

## Practical workflow

When starting a new dataset:

1. Check the dataset page for its domain, disk layout, and benchmark context.
2. Use `dronalize inspect <dataset>` to see defaults such as schema, filter rules, and map settings.
3. Use `dronalize split-support <dataset>` before deciding on a split strategy.
4. Override only the parts that matter for your project.

This keeps your config focused on project-specific choices instead of repeating built-in defaults.
