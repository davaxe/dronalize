# API Reference

<div class="section-intro" markdown="1">
This reference is organized by public import surface and then by what each part of the API is used for. Related symbols are grouped together on focused pages instead of being split mechanically into `Classes`, `Enums`, and `Functions`.
</div>

## Packages

| Package | Use it for |
| --- | --- |
| [dronalize](dronalize/index.md) | Top-level package namespace and entry-point overview |
| [dronalize.core](core/index.md) | Shared enums, scene schemas, and map graph types |
| [dronalize.datasets](datasets/index.md) | Dataset discovery, lookup, registration, and dataset descriptors |
| [dronalize.runtime](runtime/index.md) | Config resolution, planning, run state, and advanced executor APIs |
| [dronalize.processing](processing/index.md) | Processing config, filters, maps, pipeline types, and loader extension hooks |
| [dronalize.io](io/index.md) | Export config, manifests, persisted storage contracts, readers, and adapters |
| [dronalize.plot](plot/index.md) | Plotting helpers for maps and trajectories |

## Structure

- Package landing pages explain what is public from each module and link to focused pages such as `Configuration`, `Planning and runs`, or `Registry operations`.
- Obvious symbol families are documented together on the same page when they are typically used together.
- The main user-facing API is organized around package namespaces. Advanced extension hooks live on focused pages instead of being flattened into the root package.

## Related guides

- [Architecture](../../concepts/architecture.md)
- [Configuration model](../../concepts/configuration-model.md)
- [Outputs and schemas](../../concepts/outputs-and-schemas.md)
- [Dataset reference](../datasets/index.md)
