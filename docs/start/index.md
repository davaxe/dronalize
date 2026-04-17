# Start

<div class="section-intro" markdown="1">
This section covers the shortest path from installation to a real preprocessing run, then points
to the next pages you are likely to need.
</div>

## In this section

| Page | Use it for |
| --- | --- |
| [Installation](installation.md) | Install the package and add only the extras you actually need. |
| [First run (CLI)](first-run-cli.md) | Inspect a dataset, preview the resolved plan, and run the CLI once. |
| [Python entry](python-entry.md) | Resolve jobs and run preprocessing from Python. |
| [Adding datasets](adding-datasets.md) | Add a custom dataset by implementing a loader and registering a [`DatasetSpec`][dronalize.datasets.DatasetSpec]. |
| [Changelog](changelog.md) | Review user-facing package and documentation changes. |
| [Citation](citation.md) | Cite `dronalize` in academic or research work. |

## Suggested order

1. Install the base package plus the extras you need for your workflow.
2. Use `dronalize available`, `inspect`, and `show-config` to understand one dataset.
3. Run `dronalize process --plan` once before writing output.
4. Switch to the Python runtime API if you want to embed preprocessing in code.

## Related sections

- [Concepts](../concepts/index.md)
- [Configuration reference](../reference/configuration/index.md)
- [Storage backends](../outputs/index.md)
