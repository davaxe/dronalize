# Adding datasets

<div class="section-intro" markdown="1">
Custom datasets enter `dronalize` through the same registry surface as built-in datasets: a loader
turns raw files into normalized source data, and a [`DatasetSpec`][dronalize.datasets.DatasetSpec]
describes how that loader should be configured and discovered.
</div>

## Minimal path

1. Implement a [`BaseSceneLoader`][dronalize.processing.loading.BaseSceneLoader] subclass that finds
   raw sources and loads them into the shared trajectory representation.
2. Define a [`DatasetSpec`][dronalize.datasets.DatasetSpec] with a unique `name`, `loader_type`,
   `default_config`, and `native_schema`.
3. Register it before resolving or executing a request. For Python code, call
   [`dronalize.datasets.register()`][dronalize.datasets.register] directly. For CLI usage, expose the
   dataset through the module hook below.
4. Verify it with a small [`ExecutionRequest`][dronalize.runtime.ExecutionRequest] and
   [`resolve_request()`][dronalize.runtime.resolve_request], or with `process --plan` through the CLI
   once the module hook is in place.

## Registration scope

`register()` updates the in-memory registry for the current Python process. A normal shell command
such as `dronalize inspect <name>` starts a new process, so it will not see registrations performed
by unrelated external code.

That means this works for Python-driven runs:

```python
from pathlib import Path

from dronalize.datasets import register
from dronalize.runtime import ExecutionRequest, resolve_request
from my_project.datasets import MY_DATASET_SPEC

register(MY_DATASET_SPEC)

request = ExecutionRequest(
    dataset=MY_DATASET_SPEC.name,
    input_dir=Path("raw"),
    output_dir=Path("processed"),
)
plan = resolve_request(request)
```

It does not make the dataset visible to a later, separate `dronalize ...` shell command.

## CLI usage

For CLI workflows, put the dataset registration in an importable Python module and expose a
`register_dronalize_datasets()` hook. The hook can register specs itself:

```python
from dronalize.datasets import register
from my_project.datasets import MY_DATASET_SPEC


def register_dronalize_datasets():
    register(MY_DATASET_SPEC)
```

The hook may also return one [`DatasetSpec`][dronalize.datasets.DatasetSpec] or an iterable of
specs instead:

```python
from my_project.datasets import MY_DATASET_SPEC


def register_dronalize_datasets():
    return [MY_DATASET_SPEC]
```

Pass the module with `--dataset-module` before the command name:

```bash
dronalize --dataset-module my_project.dronalize_datasets available
dronalize --dataset-module my_project.dronalize_datasets inspect <name>
dronalize --dataset-module my_project.dronalize_datasets split-support <name>
dronalize --dataset-module my_project.dronalize_datasets process <name> --input raw --output processed --plan
```

The CLI imports the module and calls `register_dronalize_datasets()` before resolving dataset names,
so the same option works with `available`, `inspect`, `show-config`, `split-support`, and `process`.

Limitations:

- `--dataset-module` must be passed on each CLI invocation that needs the custom dataset.
- The module must be importable in the CLI environment, for example because the package is installed
  or its parent directory is on `PYTHONPATH`.
- The hook runs as Python code, so only use modules you trust.
- There is no automatic discovery of external dataset packages unless they are imported with
  `--dataset-module`.

## Keep the first integration small

Start with one source format, one default scene window, and no optional map handling. Add native
splits, dataset-specific options, maps, or specialized screening only after the basic loader can
produce valid scenes.
