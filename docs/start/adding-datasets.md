# Adding datasets

<div class="section-intro" markdown="1">
Custom datasets enter `dronalize` through the same registry surface as built-in datasets: a loader
turns raw files into normalized source data, and a [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor)
describes how that loader should be configured and discovered.
</div>

## Minimal path

1. Implement a [`SceneLoader`](../reference/api/processing/loading.md#dronalize.processing.loading.SceneLoader) subclass that finds
   raw sources and loads them into the shared trajectory representation.
2. Define a [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor) with a unique `name`, `loader_factory`,
   `default_config`, and `native_schema`.
3. Register it before resolving or executing a request. For Python code, call
   [`dronalize.datasets.register_dataset()`](../reference/api/datasets/registry.md#dronalize.datasets.register_dataset) directly. For CLI usage, expose the
   dataset through the module hook below.
4. Verify it with a small [`ExecutionRequest`](../reference/api/runtime/planning-and-runs.md#dronalize.runtime.ExecutionRequest) and
   [`resolve_request()`](../reference/api/runtime/executor.md#dronalize.runtime.resolve_request), or with `process --plan` through the CLI
   once the module hook is in place.

## Registration scope

`register_dataset()` updates the in-memory registry for the current Python process. A normal shell command
such as `dronalize inspect <name>` starts a new process, so it will not see registrations performed
by unrelated external code.

That means this works for Python-driven runs:

<!-- no-validate -->
```python
from pathlib import Path

from dronalize.datasets import register_dataset
from dronalize.runtime import ExecutionRequest, resolve_request
from my_project.datasets import MY_DATASET_DESCRIPTOR

register_dataset(MY_DATASET_DESCRIPTOR)

request = ExecutionRequest(
    dataset=MY_DATASET_DESCRIPTOR.name, input_dir=Path("raw"), output_dir=Path("processed")
)
plan = resolve_request(request)
```

It does not make the dataset visible to a later, separate `dronalize ...` shell command.

## CLI usage

For CLI workflows, put the dataset registration in an importable Python module and expose a
`register_dronalize_datasets()` hook. The hook can register specs itself:

<!-- no-validate -->
```python
from dronalize.datasets import register_dataset
from my_project.datasets import MY_DATASET_DESCRIPTOR


def register_dronalize_datasets():
    register_dataset(MY_DATASET_DESCRIPTOR)
```

The hook may also return one [`DatasetDescriptor`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetDescriptor) or an iterable of
specs instead:

<!-- no-validate -->
```python
from my_project.datasets import MY_DATASET_DESCRIPTOR


def register_dronalize_datasets():
    return [MY_DATASET_DESCRIPTOR]
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
splits, loader options, maps, or specialized screening only after the basic loader can produce valid
scenes.

Optional dataset capabilities are explicit. Set
[`DatasetFeatureSupport`](../reference/api/datasets/descriptor.md#dronalize.datasets.DatasetFeatureSupport)
on the spec when the loader can provide maps or lane-change sampling. Leave a flag disabled until
the loader really supports it; request planning fails early when a user enables an unsupported
feature.

Dataset-owned loader settings belong in `loader_options_model`, and users configure them under
`[datasets.<name>.loader_options]`. Avoid adding dataset-specific keys to unrelated generic config
sections.

<!-- no-validate -->
```python
from dronalize.datasets import DatasetFeatureSupport, DatasetDescriptor
from my_project.options import MyLoaderOptions

MY_DATASET_DESCRIPTOR = DatasetDescriptor(
    name="my-dataset",
    loader_factory=MyLoader.from_loader_request,
    default_config=MY_DEFAULT_CONFIG,
    native_schema=MY_NATIVE_SCHEMA,
    loader_options_model=MyLoaderOptions,
    feature_support=DatasetFeatureSupport(map=True),
)
```

## Simple example

The  `examples/custom_dataset.py` script demonstrates a minimal custom dataset integration. It is self-contained and covers both supported usage patterns:

- Python-driven dataset registration and execution
- CLI-based loading via `--dataset-module`

For example, run the CLI `inspect` command from the repository root with the custom dataset module:

```sh
PYTHONPATH=. dronalize --dataset-module examples.custom_dataset inspect mini-csv
```

!!! note "Python path"
    `PYTHONPATH=.` makes the local examples package importable when running directly from the repository checkout. In normal projects, the dataset module is usually part of an installed package, so this extra environment variable is not required.
