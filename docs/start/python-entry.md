# Python entry

<div class="section-intro" markdown="1">
The public Python surface is organized around package namespaces rather than root-level shortcuts. Start with the dataset registry, runtime config API, processing config models, and the scene or map types you actually need.
</div>

## Inspect a dataset programmatically

<div class="command-block" markdown="1">
```python
from dronalize.datasets import get

dataset_spec = get("a43")
print(dataset_spec.name)
print(dataset_spec.has_map)
print(dataset_spec.supported_split_strategies)
```
</div>

The registry returns a `DatasetSpec`, which carries loader defaults, native schema information, split support, and optional capabilities.

## Resolve a config file in Python

```python
from pathlib import Path

from dronalize.datasets import get
from dronalize.runtime.config import load_project_config, resolve_dataset_config

dataset_spec = get("a43")
project_config = load_project_config(Path("config.toml"))
resolved = resolve_dataset_config(project=project_config, descriptor=dataset_spec)

print(resolved.export.trajectory_schema.name)
print(resolved.execution.jobs)
```

This gives you the same declarative config-resolution path the CLI uses before a job is compiled for execution.

## Construct processing config directly

```python
from dronalize.processing import LoaderConfig, MapConfig

loader = LoaderConfig(history_frames=20, future_frames=30, sample_time=0.1)
map_config = MapConfig.default()
```

## Scene-schema helpers

The `dronalize.core.scene` package exports built-in schema constants and lookup helpers such as `CANONICAL`, `POSITIONS_VELOCITY_YAW`, and `get_trajectory_schema`. Map graph types live separately in `dronalize.core.maps`.
