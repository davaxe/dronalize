# Python entry

<div class="section-intro" markdown="1">
The public Python surface is still intentionally compact. For this first pass, the stable entry points worth discovering are the dataset registry, runtime config API, and scene-schema helpers exported from package-level modules.
</div>

## Inspect a dataset programmatically

<div class="command-block" markdown="1">
```python
from dronalize.datasets import get

descriptor = get("a43")
print(descriptor.name)
print(descriptor.has_map)
print(descriptor.supported_split_strategies)
```
</div>

The registry returns a `DatasetDescriptor`, which carries loader defaults, native schema information, split support, and optional capabilities.

## Resolve a config file in Python

```python
from pathlib import Path

from dronalize.datasets import get
from dronalize.runtime import ConfigResolver, load_project_config

descriptor = get("a43")
file_config = load_project_config(Path("config.toml"))
resolved = ConfigResolver().resolve(descriptor=descriptor, file_config=file_config)

print(resolved.writer.scene_schema.name)
print(resolved.execution.jobs)
```

This gives you the same resolver path the CLI uses when it combines descriptor defaults with the authoring config file.

## Scene-schema helpers

The `dronalize.core.scene` package exports built-in schema constants and lookup helpers such as `CANONICAL`, `POSITIONS_VELOCITY_YAW`, and `get_scene_schema`.
