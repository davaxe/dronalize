# `[runtime]` section

<div class="section-intro" markdown="1">
Runtime settings control how work is distributed while processing. In most cases, this section is only about how many workers to use and whether work should be batched.
</div>

| Key | Type | Description | Default |
|---|---|---|---|
| `jobs` | `int` or `"auto"` | Number of worker processes to use. Set `"auto"` to let the runtime choose automatically. | `1` |
| `chunksize` | `int` | Number of work items batched together per executor dispatch. | `none` |

`jobs = 1` keeps execution serial. Any value greater than `1`, or `"auto"`, enables the parallel executor.

!!! note "Validation"
    `jobs` must be a positive integer or `"auto"`.

## Minimal example

```toml

# Define a profile named "basic" with runtime settings.
[profiles.basic.runtime]
jobs = "auto"

# Specify runtime settings for a43 dataset.
[datasets.a43.runtime]
jobs = 4
chunksize = 64
```
