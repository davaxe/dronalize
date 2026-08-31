# API reference

The public Python API is organized around shallow package namespaces. Prefer these imports in user
code:

```python
from prejectory.config import parse_config
from prejectory.core import Scene, MapGraph, AgentCategory, get_trajectory_schema
from prejectory.datasets import get_dataset, list_datasets
from prejectory.runtime import ExecutionRequest, execute_request, resolve_request
from prejectory.io import SceneRecord, read_manifest
from prejectory.io.readers import PickleReader
```

The pages in this section document symbols intended for direct use. Internal runtime executors,
writer backends, generated protobuf modules, and dataset-specific loader implementation classes are
not part of the curated user-facing API reference.
