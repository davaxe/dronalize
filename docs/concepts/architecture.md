# Architecture

<div class="section-intro" markdown="1">
`dronalize` is built around one idea: datasets stay dataset-specific at the edges, while the middle
of the runtime stays shared. A run resolves configuration, opens one dataset loader, turns raw
sources into normalized scenes, and hands those scenes to a storage backend.
</div>

<figure markdown="block" class="full-figure">
  ![Design architecture](../assets/architecture-dark.svg#only-dark){ width="100%" }
  ![Design architecture](../assets/architecture-light.svg#only-light){ width="100%" }
  <figcaption>
    High-level architecture diagram. The CLI and Python API both resolve the same runtime plan and
    execute the same loader, scene-building, and writing flow.
  </figcaption>
</figure>

## Main runtime objects

Four public objects define most of the runtime:

- [`DatasetSpec`][dronalize.datasets.DatasetSpec] describes one dataset integration
- [`ProcessingConfig`][dronalize.config.ProcessingConfig] represents the
  optional TOML file
- [`ExecutionRequest`][dronalize.runtime.ExecutionRequest] describes one
  requested run
- [`ExecutionPlan`][dronalize.runtime.ExecutionPlan] is the execution-ready
  result after config resolution and planning

The CLI is a thin layer that builds an
[`ExecutionRequest`][dronalize.runtime.ExecutionRequest], resolves an
[`ExecutionPlan`][dronalize.runtime.ExecutionPlan], and then either prints it or
executes it.

## Resolution flow

Every run starts by looking up a dataset key in the registry. That returns a
[`DatasetSpec`][dronalize.datasets.DatasetSpec] with the dataset defaults,
native schema, native split support, map support, and any dataset-owned config
model.

The runtime then resolves configuration in this order:

1. the dataset's built-in `default_config`
2. any profiles named in `[datasets.<name>].uses`
3. the dataset-local TOML entry in `[datasets.<name>]`
4. runtime overrides from the CLI or
   [`RuntimeOverride`][dronalize.config.RuntimeOverride]

That produces one resolved [`DatasetConfig`][dronalize.config.models.DatasetConfig].
The runtime compiles it into narrower subsystem plans:

- a loader-facing [`LoaderRequest`][dronalize.processing.models.LoaderRequest]
- an `OutputPlan`
- effective scene metrics after any resampling

The result is an [`ExecutionPlan`][dronalize.runtime.ExecutionPlan], which is
what [`resolve_request()`][dronalize.runtime.resolve_request],
[`execute_request()`][dronalize.runtime.execute_request], and
[`execute_plan()`][dronalize.runtime.execute_plan] work with.

## Dataset boundary

The dataset boundary is the loader API in `dronalize.processing.loading`.

A loader is responsible for:

- discovering raw sources
- reading one source into one or more Polars `LazyFrame` objects
- attaching lightweight map bindings when needed
- exposing dataset-specific options and map resolution behavior

The shared runtime is responsible for everything after that:

- screening
- split routing
- resampling
- scene numbering
- schema conversion
- backend writing

## Execution flow

After planning, execution does three things:

1. opens any run-scoped shared resources declared by the
   [`DatasetSpec`][dronalize.datasets.DatasetSpec]
2. builds the loader
3. chooses a sequential or parallel executor

Execution then follows two nested loops.

### Source loop

The loader yields [`Source`][dronalize.processing.loading.Source] objects. A
source is a stable unit of raw input, usually a file or a directory-backed
sample group. If the dataset has native splits, the runtime can iterate those
partitions directly. Otherwise it uses `discover_sources()`.

For each source, the loader returns one or more
[`LoadedSourceData`][dronalize.processing.loading.LoadedSourceData] objects.
Each one contains:

- a Polars `LazyFrame`
- an optional [`MapBinding`][dronalize.processing.loading.MapBinding]
- an optional predefined split

### Scene loop

The loader's pipeline is built once and reused. The default pipeline can apply:

- time partitioning for `time` and `shuffled-time` splits
- sliding-window extraction
- screening
- resampling
- final grouping into one scene frame at a time

The scene-building step then turns each prepared frame into a
[`Scene`][dronalize.core.scene.Scene] by:

1. assigning a split
2. attaching a lazy map resolver when maps are enabled
3. converting the scene to the requested output schema if needed

The split assignment strategy depends on the configured mode:

- `native` uses the dataset's predefined partitions
- `scene` hashes the stable per-source scene identifier
- `source` hashes the source identifier
- `time` and `shuffled-time` read the partition column produced earlier in the pipeline

## Output flow

Each final [`Scene`][dronalize.core.scene.Scene] is encoded into the shared
record layout and passed to a dataset writer. The writer comes from the backend
registry in `dronalize.io.backends`.

The built-in backends are:

- `pickle` for one pickled [`SceneRecord`][dronalize.io.SceneRecord] per scene
- `mds` for Mosaic Streaming shards
- `null` for runs that should execute but not persist scenes

Regardless of backend, the runtime writes one `manifest.json` at the dataset root. That manifest is
the stable description of the produced dataset: schema, feature columns, sample time, derived
features, map presence, and precision.

## Module map

| Module | Responsibility |
| --- | --- |
| `dronalize.datasets` | Dataset registry, specs, and built-in dataset definitions |
| `dronalize.config` | TOML loading and runtime override helpers |
| `dronalize.runtime` | Request models, planning, execution, and the CLI |
| `dronalize.processing` | Loader API, pipeline assembly, screening, resampling, and map helpers |
| `dronalize.core` | Scene types, map graph types, schemas, categories, and shared errors |
| `dronalize.io` | Encoding, manifests, backends, readers, and optional adapters |
