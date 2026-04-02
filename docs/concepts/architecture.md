# Architecture

<div class="section-intro" markdown="1">
The architecture of `dronalize` is designed to be modular, flexible, and extensible.
</div>


<figure markdown="block" class="full-figure">
  ![Design architecture](../assets/architecture-dark.svg#only-dark){ width="100%" }
  ![Design architecture](../assets/architecture-light.svg#only-light){ width="100%" }
  <figcaption>
    High level architecture diagram. The <code>dronalize</code> library provides the main API and processing logic, while configuration files define the specific behavior for each run. The CLI is a thin wrapper around the core library for ease of use.
  </figcaption>
</figure>

## How the pieces fit together

Three things determine what a run does: the **dataset** selected, the **configuration** applied, and the **output** requested. The diagram captures these as three horizontal phases that always execute in the same order.

Before any data is touched, a config resolution step merges dataset defaults, a TOML file, and CLI flags into a single resolved plan. That plan is then used to open a loader, assemble the processing pipeline, and prepare the writer. The actual data processing follows two nested loops: a **source loop** that iterates over raw files or recordings, and a **scene loop** inside it that handles each scene window produced from a source.

## Config resolution

Every run starts from the dataset's built-in defaults. The TOML config file can override them, first through a `[global]` block that applies everywhere, then through a `[datasets.<name>]` block that applies only to one dataset. CLI flags are applied last.

The `ConfigResolver` in `dronalize.runtime` handles this merge and produces a single resolved config used for the rest of the run. The same resolver is used whether the run is started from the CLI or from Python via `plan_dataset()`.

!!! note "Dataset defaults as the foundation"
    Many values — such as `input_len`, `output_len`, and `sample_time` — come from the
    dataset's own built-in loader config, not from a package-level constant. Omitting a field
    from your config file leaves the dataset's defaults intact. Check the [dataset reference](../reference/datasets/index.md) or use `dronalize inspect <dataset>` to see what a dataset starts with.

For a full description of the layering rules and override precedence, see the [configuration model](configuration-model.md).

## Dataset registry

Each built-in dataset is described by a `DatasetDescriptor` — a small object that carries everything needed to process it: the loader factory, default loader and map configs, the native scene schema, and the split strategies it supports.

The registry is lazy: a dataset's module is only imported when it is first requested by name. This means optional dependencies for datasets you are not using — such as `protobuf` for Waymo or `zarr` for Lyft — do not need to be installed.

Descriptors can be explored from Python via `dronalize.datasets.get()` or from the CLI with `dronalize inspect <dataset>` and `dronalize split-support <dataset>`.

## The processing pipeline

Once a plan is resolved and a run is opened, processing follows the two loops shown in the diagram.

### Source loop

The loader discovers raw input sources — usually files or directories — and iterates over them one at a time. For datasets with predefined splits such as Argoverse 2 or Waymo, sources are drawn from the relevant partition directories. For others, all files are discovered together and split assignment happens further down the pipeline.

Each source is read into a Polars `LazyFrame` and passed through a composable `Pipeline` — a chain of lazy transforms built once per run from the resolved loader config. The pipeline steps that are active depend entirely on what was configured:

| Step | What it does | Conditional? |
| --- | --- | --- |
| Temporal windowing | Slides a fixed-length window over the source, producing multiple overlapping scene candidates | Only when `window` is configured |
| Filtering | Removes rows, rejects scenes failing quality rules, and prunes invalid agents | Only when `filter` is configured |
| Resampling | Interpolates trajectories to a different temporal resolution | Only when `resampling` is configured |
| Scene grouping | Partitions the processed frame into one DataFrame per scene | Always |

Steps not present in the config are skipped entirely, so the pipeline is always the minimal one the current run requires. The `Pipeline` itself is immutable and composable — each active step appends a transform to the chain, and the chain is executed lazily until scene grouping collects the results.

### Scene loop

Scene grouping hands off individual DataFrames to the scene loop, where each scene is finalized before writing:

1. **Split assignment** — determines whether the scene belongs to `train`, `val`, or `test`. The mechanism depends on the split mode: time-based modes read a column written earlier in the pipeline, scene-based mode uses a stable hash of the scene identifier, and native or source modes use the partition already assigned to the source.
2. **Map resolution** — attaches the map graph to the scene if the dataset supports it and maps are enabled. The graph is not materialized until it is needed for encoding.
3. **Scene construction and schema conversion** — wraps the DataFrame in a `Scene` object. If the configured output schema requests fields not present in the dataset's native schema — for example, velocity from a positions-only source — those fields are derived here.

## Output

Each finalized `Scene` is encoded into a dictionary of NumPy arrays and handed to a `SceneWriter`. The encoding step produces a dense `[agents, timesteps, features]` tensor, a matching presence mask, and optionally recenters positions around the scene mean when `offset_positions` is enabled.

The current stable output format is MDS (Mosaic Streaming), which writes binary shards with a per-split `index.json`. A `manifest.json` is written alongside the shards and records the schema name, feature columns, sequence lengths, precision, and whether map data is present. This manifest is the stable description of a processed dataset for downstream consumers.

The `dummy` writer skips all file I/O and is useful for validating a config or benchmarking the pipeline without writing output. See [outputs and schemas](outputs-and-schemas.md) for more on schema choice and writer options.

## Module map

| Module | Responsibility |
| --- | --- |
| `dronalize.datasets` | Dataset registry, descriptors, and built-in dataset definitions |
| `dronalize.runtime` | Config resolution, execution planning, CLI, and run orchestration |
| `dronalize.processing` | Pipeline, filters, resampling, map processing, and loader base classes |
| `dronalize.core` | Scene data model, schema definitions, agent categories, and shared error types |
| `dronalize.io` | Writers, output encoding, manifest I/O, and storage protocols |
| `dronalize.plot` | Optional visualization helpers for trajectories and map graphs |
