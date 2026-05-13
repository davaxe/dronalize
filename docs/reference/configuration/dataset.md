# `[loader_options]` section

`loader_options` is a dataset-specific table for dataset-owned loader parameters. It is validated against the selected dataset's typed loader options model and is only valid for datasets that expose dataset-specific config.

Example for a dataset that supports loader options:

```toml
[datasets.argoverse1.loader_options]
file_batch_size = 8
```

Use `dronalize inspect <dataset>` to see whether a dataset exposes loader options and which option keys it supports by default. The [dataset reference](../datasets/index.md) also documents dataset-specific config when relevant.
