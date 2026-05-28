# Architecture overview

<div class="section-intro" markdown="1">
`dronalize` is built around a simple architectural idea: dataset-specific logic stays at the edges, while the runtime in the middle stays shared. A run resolves its configuration, opens a dataset loader, turns raw sources into normalized scenes, and writes those scenes through a storage backend.

This keeps dataset integrations focused on ingestion and metadata, while the rest of the system can stay consistent across all datasets.
</div>

<figure markdown="block" class="full-figure">
  ![Design architecture](../assets/architecture-dark.svg#only-dark){ width="100%" }
  ![Design architecture](../assets/architecture-light.svg#only-light){ width="100%" }
  <figcaption>
    High-level architecture diagram. Dataset-specific logic stays near source loading and map access, while planning, scene construction, and output writing remain shared.
  </figcaption>
</figure>

## The basic shape of a run

At a high level, every run follows the same path:

1. the CLI or Python API receives a request
2. the runtime resolves that request into an execution plan
3. a dataset loader exposes raw sources
4. the shared runtime turns those sources into scenes
5. a backend writes the final dataset artifacts

That separation is the main design choice in the project. Dataset code is responsible for understanding how raw data is laid out and how it should be read. Once that data has been loaded, the rest of the flow is largely dataset-agnostic.

## Why the system is organized this way

The goal is not just code reuse. It is to keep the responsibilities of the system clear.

A dataset integration should be relatively narrow: it should know how to discover sources, load them, and expose any dataset-native metadata such as map references or predefined partitions. The shared runtime should own the common execution flow: planning, scene construction, split assignment, schema conversion, and writing output.

This makes new dataset integrations easier to add without duplicating the rest of the runtime. It also makes the processing model easier to reason about, because the same overall flow applies regardless of which dataset is being converted.

## Main runtime concepts

A few public types define that flow. [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor) describes one dataset integration. [`ProjectConfig`](../reference/api/config/index.md#dronalize.config.ProjectConfig) represents optional user configuration. [`ExecutionRequest`](../reference/api/runtime/planning-and-runs.md#dronalize.runtime.ExecutionRequest) describes the requested run, and [`ExecutionPlan`](../reference/api/runtime/planning-and-runs.md#dronalize.runtime.ExecutionPlan) is the resolved, execution-ready result.

Together, these types separate what the user asked for from what the runtime is actually going to execute.

## Output model

The final output is always the same in concept:

- a collection of normalized scenes
- encoded in a backend-specific format
- plus a manifest describing what was produced

Different backends can store those scenes in different formats, but that choice does not change the runtime model itself. Writing is treated as the last stage of a shared pipeline, not as something that reshapes the rest of the system.

## Module guide

| Module | Role |
| --- | --- |
| `dronalize.datasets` | Dataset definitions, registry, and dataset-specific integration points |
| `dronalize.config` | User-facing configuration and override handling |
| `dronalize.runtime` | Requests, planning, execution, and CLI entry points |
| `dronalize.processing` | source loading, shared processing flow, scene preparation, and map helpers |
| `dronalize.core` | Shared domain types such as scenes, schemas, maps, and common errors |
| `dronalize.io` | Encoding, manifests, storage backends, and dataset readers |
