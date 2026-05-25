# Python entry

<div class="section-intro" markdown="1">
The Python surface is organized around explicit package namespaces. In practice, most code starts
with `dronalize.datasets`, `dronalize.config`, and `dronalize.runtime`.
</div>

## Inspect a dataset programmatically

```python
from dronalize.datasets import get_dataset

spec = get_dataset("a43")
print(spec.name)
print(spec.feature_support.map)
print(spec.feature_support.lane_change_sampling)
print(spec.loader_options_model.model_fields)
print(spec.native_schema.name)
print(spec.supported_native_splits)
```

`get_dataset()` returns a `DatasetDescriptor`, which is the same descriptor the CLI uses for `inspect` and
`split-support`.

## Resolve a config file

<!-- no-validate -->
```python
from pathlib import Path

from dronalize.config import parse_config
from dronalize.datasets import get_dataset

spec = get_dataset("a43")
project = parse_config(Path("dronalize.toml"))
resolved = project.resolve_dataset_config("a43", spec.default_config)

print(resolved.scenes.horizon_frames, resolved.scenes.default_observation_length)
print(resolved.output.precision)
print(resolved.read.root.strategy)
print(resolved.assign.root.strategy)
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
print(result.selected_scenes)
```

Use `resolve_request()` when you want a dry planning step. Use `execute_request()` when you want to
execute the request directly.

## Schema and record helpers

The `dronalize.core.scene` package exports the built-in schema constants and lookup helpers such as
`CANONICAL`, `POSITIONS_VELOCITY_YAW`, and `get_trajectory_schema`.

For persisted outputs:

- `dronalize.io.readers` provides framework-neutral readers
- `dronalize.io.adapters` provides optional Torch and PyG dataset adapters
