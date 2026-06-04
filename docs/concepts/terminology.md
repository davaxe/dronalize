# Terminology

This page defines the terms used across `dronalize` docs, configuration, and public APIs.

## Dataset and loading

| Term | Definition |
| --- | --- |
| Dataset | A named raw trajectory corpus that `dronalize` can read, such as `a43` or `waymo`. |
| Dataset integration | The complete dataset-specific implementation: descriptor, loader, defaults, optional maps, and any supporting helpers. |
| Dataset descriptor | The registry object, [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor), that connects a dataset key to its loader, defaults, schemas, capabilities, and metadata. |
| Loader | A `SceneLoader` implementation that discovers raw inputs and emits source frames for the shared runtime. |
| Source | One raw input unit exposed by a loader, usually a file, recording, scenario, shard, or query result. |
| Source frame | A lazy DataFrame emitted by a loader from one source before shared scene-window extraction and screening. |

## Scene lifecycle

| Term | Definition |
| --- | --- |
| SceneCandidate | A candidate scene window produced before final acceptance. It carries a DataFrame, screening result, source metadata, map reference, and possible output split assignment. |
| Scene | The internal DataFrame-backed runtime object used during processing and writing. |
| SceneRecord | The dense persisted and reader-facing representation. Writers encode `Scene` objects into `SceneRecord` unless an explicit output transform bypasses record encoding. |
| Scene record | A serialized or in-memory `SceneRecord` payload. |

## Splits

| Term | Definition |
| --- | --- |
| Native split | An upstream dataset partition such as train, validation, or test. Native splits come from the source dataset. |
| Read selection | The raw inputs selected for reading, for example all sources or only a native split. |
| Output split | The final train/val/test routing used when writing processed scene records. |
| Unsplit output | Output written without train/val/test assignment. It is stored under the `unsplit` directory. |
| Source split | A native/source split carried by a discovered source before output split assignment. |

## Time

| Term | Definition |
| --- | --- |
| Horizon | The number of timesteps in a scene window or scene record. |
| Configured horizon | The scene-window length requested before optional resampling. |
| Effective horizon | The horizon after optional resampling is applied. |
| Persisted horizon | The actual stored timestep count in each `SceneRecord`. |
| Observation length | The history length used when splitting a full horizon into observation/history and prediction/future portions. |
| `sample_time` | The time interval between adjacent frames or timesteps, in seconds. |

## Output and maps

| Term | Definition |
| --- | --- |
| Backend | The storage implementation used to write scene records, such as `pickle`, `mds`, or `null`. |
| Manifest | The `manifest.json` file that records dataset names, trajectory schema, feature fields, timing, horizon, precision, backend, and map presence for an export. |
| Map reference | A lightweight `MapReference` carried by loader output to identify or provide map data for a scene. |

## Schemas and encoded labels

| Term | Definition |
| --- | --- |
| Trajectory schema | The canonical concept for trajectory fields and feature conversion. In TOML, `output.schema` is the concise config key for the output trajectory schema. |
| `agent_category` | Semantic DataFrame column containing the agent category before encoding. |
| `agent_types` | Encoded per-agent integer array in `SceneRecord`. |
| `EdgeType` and map node types | Semantic map labels used while constructing map graphs. |
| `map_edge_types` and `map_node_types` | Encoded arrays stored in `SceneRecord`. |
