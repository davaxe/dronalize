# API Reference

<div class="section-intro" markdown="1">
This reference is organized by public import surface and then by what each part of the API is used for. Related symbols are grouped together on focused pages instead of being split mechanically into `Classes`, `Enums`, and `Functions`.
</div>

## Packages

| Package | Use it for |
| --- | --- |
| [dronalize](dronalize/index.md) | Root-level shortcuts for common configuration and scene access |
| [dronalize.core](core/index.md) | Shared categories and split enums |
| [dronalize.datasets](datasets/index.md) | Dataset discovery, lookup, and descriptor models |
| [dronalize.runtime](runtime/index.md) | Config resolution, planning, summaries, and run state |
| [dronalize.io](io/index.md) | Writer config, manifests, and persisted storage contracts |
| [dronalize.plot](plot/index.md) | Plotting helpers for maps and trajectories |

## Structure

- Package landing pages explain what is public from each module and link to focused pages such as `Configuration`, `Planning and runs`, or `Registry operations`.
- Obvious symbol families are documented together on the same page when they are typically used together.
- Only the curated public surface is documented here. Deeper implementation modules are intentionally omitted unless they are re-exported publicly.

## Related guides

- [Architecture](../../concepts/architecture.md)
- [Configuration model](../../concepts/configuration-model.md)
- [Outputs and schemas](../../concepts/outputs-and-schemas.md)
- [Dataset reference](../datasets/index.md)
