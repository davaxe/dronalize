# Concepts

<div class="section-intro" markdown="1">
This section explains the processing model behind `dronalize`: how configuration is resolved, how
scenes are built, how splits and filters behave, and how outputs are written.
</div>

## In this section

| Page | Use it for |
| --- | --- |
| [Architecture](architecture.md) | Understand the main runtime flow from config resolution to output writing. |
| [Configuration model](configuration-model.md) | Learn how dataset defaults, global config, and dataset-specific overrides combine. |
| [Split strategies](split-strategies.md) | Compare the available split modes and when to use them. |
| [Filtering](filtering.md) | See how scene and agent-level filtering rules are applied. |
| [Map processing](map-processing.md) | Understand map extraction and map graph handling. |
| [Datasets](datasets.md) | See how built-in dataset support is organized. |
| [Outputs and schemas](outputs-and-schemas.md) | Learn how scene schemas and writer outputs fit together. |
| [Highway pipeline](highway.md) | Follow a concrete end-to-end example for one dataset family. |

## Reading path

Start with [Architecture](architecture.md), then [Configuration model](configuration-model.md),
and then whichever page matches the part of the pipeline you are changing or trying to understand.

## Related sections

- [Start](../start/index.md)
- [Configuration reference](../reference/configuration/index.md)
- [Dataset reference](../reference/datasets/index.md)
