# API reference

The public Python API is organized around shallow package namespaces. Prefer these imports in user
code:

```python
from prejectory import ExecutionRequest, plan, run, open_dataset
from prejectory.config import DatasetConfigPatch, ProjectConfig, parse_config
from prejectory.core import Scene, MapGraph, AgentCategory, get_trajectory_schema
from prejectory.datasets import get_dataset, list_datasets
from prejectory.io import SceneRecord, ForecastRecord, PredictionBounds, read_manifest
from prejectory.io.readers import PickleReader
```

The pages in this section document symbols intended for direct use. Internal runtime executors,
writer backends, generated protobuf modules, and dataset-specific loader implementation classes are
not part of the curated user-facing API reference.

Configuration models have one canonical home in `prejectory.config`. Optional readers and adapters
load their dependencies lazily. Underscore-prefixed plan fields and mutable runtime state are internal.
