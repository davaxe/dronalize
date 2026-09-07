# Python entry

Start with `ExecutionRequest`, `plan`, and `run`. Planning resolves configuration without writing
output; running a plan executes exactly that resolved configuration.

<!-- no-validate -->
```python
from pathlib import Path
from prejectory import ExecutionRequest, open_dataset, plan, run
from prejectory.config import DatasetConfigPatch, OutputPatch

request = ExecutionRequest(
    dataset="a43",
    input_dir=Path("data/a43/raw"),
    output_dir=Path("data/a43/processed"),
    overrides=DatasetConfigPatch(output=OutputPatch(precision="float64")),
)
resolved = plan(request)
print(resolved.summary())
print(resolved.diagnostics)

result = run(resolved)
print(result.written_scenes)
dataset = open_dataset(result.output_dir)
print(dataset.manifest.feature_columns)
```

Use `run(request)` when you do not need to inspect a plan first. The explicit runtime functions
`resolve_request`, `execute_request`, and `execute_plan` remain available. Plans expose `config`,
absolute `input_dir`/`output_dir`, `trajectory_schema`, `include_map`, and effective timing.
Compiled loader, descriptor, and assignment state is internal. To change a plan, change its request
and resolve again.

## Configure runs in memory

`config` accepts a `ProjectConfig`, a TOML path (`str` or `Path`), or `None`. The same resolution
machinery handles Python and TOML: dataset defaults, project defaults/profiles, dataset entries,
and finally `overrides: DatasetConfigPatch`.

```python
from prejectory.config import (
    Clear,
    DatasetConfigEntry,
    DatasetConfigPatch,
    MapPatch,
    OutputPatch,
    ProjectConfig,
    ScenesPatch,
    WindowPatch,
)

project = ProjectConfig(
    defaults=DatasetConfigEntry(output=OutputPatch(precision="float64")),
    datasets={"a43": DatasetConfigEntry(scenes=ScenesPatch(window=WindowPatch(step=10)))},
)
resolved_config = project.resolve_dataset_config("a43")

patch = DatasetConfigPatch(
    map=MapPatch(min_distance=None),
    screening=Clear(),
)
```

Patches preserve unspecified nested defaults. An explicit `None` sets a nullable scalar to `None`,
such as `MapPatch(min_distance=None)`. Use `Clear()` to remove an inherited optional block, such as
screening or windowing. All patch models and common configuration models are exported from
`prejectory.config`.

Select a task directly on `ExecutionRequest`: omit `task` to inherit, pass a task name or
`PredictionTaskConfig` to replace it, or pass `task=None` to disable prediction bounds. This explicit
selection takes precedence over the project's task and `overrides.task`. Configuration task bounds
use source frames; the plan and records expose bounds in stored frames after resampling.

## Inspect available datasets

```python
from prejectory.datasets import get_dataset

descriptor = get_dataset("a43")
print(descriptor.name)
print(descriptor.feature_support.map)
print(descriptor.split_support)
print(descriptor.loader_options_model.model_fields)
print(descriptor.native_schema.name)
print(descriptor.supported_native_splits)
```

`ProjectConfig.resolve_dataset_config()` accepts either a dataset name or a custom descriptor,
including its defaults and named task definitions.

## Output replacement and progress

Non-empty output directories are rejected unless `overwrite=True`. Runs write to a temporary sibling
directory and publish only after processing, writer finalization, and manifest creation succeed.
Failed processing preserves the previous output. Replacement needs disk space for both copies;
publication uses renames with rollback, rather than a crash-atomic directory exchange.

Python runs are quiet by default. Pass `show_progress=True` for Rich output or
`on_progress=callback` for `prejectory.runtime.Progress` snapshots. The callback runs on one background
thread, including its final snapshot. It must be thread-safe and should return promptly. Callback
errors propagate after workers stop and prevent output publication. Snapshots describe processing;
a returned `ExecutionResult` confirms publication. `result.stats` contains the full counters;
`result.written_scenes` is a convenience property.

Library exception types are available from `prejectory.errors`. Normal filesystem errors retain
standard Python exception types.
