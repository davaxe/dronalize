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
from dronalize.runtime import ConfigResolver, load_project_config

dataset_spec = get("a43")
file_config = load_project_config(Path("config.toml"))
resolved = ConfigResolver().resolve(descriptor=dataset_spec, file_config=file_config)

print(resolved.export.trajectory_schema.name)
print(resolved.execution.jobs)
```

This gives you the same resolver path the CLI uses when it combines dataset-spec defaults with the authoring config file.

## Construct processing config directly

```python
from dronalize.processing import LoaderConfig, MapConfig

loader = LoaderConfig(input_len=20, output_len=30, sample_time=0.1)
map_config = MapConfig.default()
```

## Scene-schema helpers

The `dronalize.core.scene` package exports built-in schema constants and lookup helpers such as `CANONICAL`, `POSITIONS_VELOCITY_YAW`, and `get_trajectory_schema`. Map graph types live separately in `dronalize.core.maps`.
