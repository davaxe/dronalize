# Python entry

The Python surface is organized around explicit package namespaces. In practice, most code starts
with `dronalize.datasets`, `dronalize.config`, and `dronalize.runtime`.

## Inspect a dataset programmatically

```python
from dronalize.datasets import get_dataset

descriptor = get_dataset("a43")
print(descriptor.name)
print(descriptor.feature_support.map)
print(descriptor.feature_support.lane_change_sampling)
print(descriptor.loader_options_model.model_fields)
print(descriptor.native_schema.name)
print(descriptor.supported_native_splits)
```

`get_dataset()` returns a `DatasetDescriptor`, which is the same descriptor the CLI uses for `inspect` and
`split-support`.

## Resolve a config file

<!-- no-validate -->
```python
from pathlib import Path

from dronalize.config import parse_config
from dronalize.datasets import get_dataset

descriptor = get_dataset("a43")
project = parse_config(Path("dronalize.toml"))
resolved = project.resolve_dataset_config(
    "a43",
    descriptor.default_config,
    named_tasks=descriptor.tasks,
    default_task=descriptor.default_task,
)

print(resolved.scenes.horizon_frames, resolved.task)
print(resolved.output.precision)
print(resolved.read.strategy)
print(resolved.assign.strategy)
```

Use `resolve_dataset_config()` when you want the final dataset config with built-in defaults
applied.

## Plan or run a request

<!-- no-validate -->
```python
from pathlib import Path

from dronalize.runtime import ExecutionRequest, execute_request, resolve_request

request = ExecutionRequest(
    dataset="a43",
    input_dir=Path("data/a43/raw"),
    output_dir=Path("data/a43/processed"),
    storage_backend="pickle",
)

plan = resolve_request(request)
print(plan.effective_sample_time)

result = execute_request(request)
print(result.written_scenes)
```

Execution rejects a non-empty output directory by default. Set
`overwrite=True` on `ExecutionRequest` only when the existing output should be
removed explicitly.

Use `resolve_request()` when you want a dry planning step. Use `execute_request()` when you want to
execute the request directly.

## Schema and record helpers

The `dronalize.core` package exports the scene, map, schema, and category types commonly used by
downstream code:

```python
from dronalize.core import CANONICAL, MapGraph, Scene, get_trajectory_schema
```

For persisted outputs:

- `dronalize.io.readers` provides framework-neutral readers
- `dronalize.io.adapters` provides optional Torch and PyG dataset adapters
