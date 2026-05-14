# Concepts

<div class="section-intro" markdown="1">
This section explains the current processing model behind `dronalize`: how runs are resolved, how
scenes are built, how screening and splits behave, and how outputs are written.
</div>

## In this section

| Page | Use it for |
| --- | --- |
| [Architecture](architecture.md) | Understand the runtime flow from dataset lookup to backend writing. |
| [Configuration model](configuration-model.md) | Learn how dataset defaults, profiles, dataset entries, and runtime overrides combine. |
| [Datasets](datasets.md) | See what a dataset key means and what a [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor) provides. |
| [Split strategies](split-strategies.md) | Compare the supported split strategies and when to use them. |
| [Screening](screening.md) | Understand cleanup, scene checks, and agent checks in the current config model. |
| [Map processing](map-processing.md) | Understand map inclusion, extraction, and geometry controls. |
| [Outputs and schemas](../formats/schema.md) | Separate schema choice from backend choice. |
| [Lane-change sampling](lane-change-sampling.md) | Follow the current lane-change-aware window selection model. |

## Related sections

- [Start](../start/index.md)
- [Configuration reference](../reference/configuration/index.md)
- [Dataset reference](../reference/datasets/index.md)
