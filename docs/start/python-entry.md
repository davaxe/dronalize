# Python entry

<div class="section-intro" markdown="1">
The Python surface is organized around explicit package namespaces. In practice, most code starts
with `dronalize.datasets`, `dronalize.config`, and `dronalize.runtime`.
</div>

## Inspect a dataset programmatically

```python
from dronalize.datasets import get

spec = get("a43")
print(spec.name)
print(spec.has_map)
print(spec.native_schema.name)
print(spec.supported_native_splits)
```

`get()` returns a `DatasetSpec`, which is the same descriptor the CLI uses for `inspect` and
`split-support`.

## Resolve a config file

```python
from pathlib import Path

from dronalize.config import load_project_config
from dronalize.datasets import get

spec = get("a43")
project = load_project_config(Path("config.toml"))
resolved = project.resolve("a43", spec.default_config)

print(resolved.scenes.history_frames, resolved.scenes.future_frames)
print(resolved.output.precision)
print(resolved.read.root.strategy)
print(resolved.assign.root.strategy)
```

Use `resolve()` when you want the final dataset config with built-in defaults applied. The lower
level `extract()` helper only returns the authored dataset entry from the file, not the fully merged
result.

## Plan or run a request

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
