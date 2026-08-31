# `[loader_options]` section

When a descriptor defines `default_task`, that named prediction task is selected automatically.
The explicit form below is equivalent and can be useful in shared config files:

```toml
[datasets.argoverse1]
task = "benchmark"
```

Set `task` to another available name to replace the default selection, or use `task = "none"` for
task-free output. Run `prejectory inspect <dataset>` to list named tasks and their effective bounds.

An inline task is a complete custom replacement, not a patch over the named task:

```toml
[datasets.argoverse1.task]
prediction_origin = 10
prediction_end = 30
```

Both bounds are required. Named and inline forms are mutually exclusive because TOML cannot define
the same `task` key as both a string and a table.

`loader_options` is a dataset-specific table for dataset-owned loader parameters. It is validated against the selected dataset's typed loader options model and is only valid for datasets that expose dataset-specific config.

Example for a dataset that supports loader options:

```toml
[datasets.argoverse1.loader_options]
file_batch_size = 8
```

Use `prejectory inspect <dataset>` to see whether a dataset exposes loader options and which option keys it supports by default. The [dataset reference](../datasets/) also documents dataset-specific config when relevant.
