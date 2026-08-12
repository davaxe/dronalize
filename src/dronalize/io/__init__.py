"""Persisted storage contracts and export-facing configuration.

## Import guide

```python
from dronalize.io import (
    DatasetManifest,
    RecordTransform,
    SceneRecord,
    SceneTransform,
    SplitSceneRecord,
    read_manifest,
    write_manifest,
)
```

Output configuration models live under [`dronalize.config.models`][] so the
runtime and CLI share one canonical configuration surface.

## Related modules

- [`dronalize.io.adapters`][] for optional Torch and PyG adapter layers
"""

from dronalize.io.base import RecordTransform, SceneTransform, StorageBackend
from dronalize.io.manifest import (
    DatasetManifest,
    PredictionTaskManifest,
    manifest_path,
    read_manifest,
    write_manifest,
)
from dronalize.io.records import PredictionBounds, SceneRecord, SplitSceneRecord

__all__ = [
    "DatasetManifest",
    "PredictionBounds",
    "PredictionTaskManifest",
    "RecordTransform",
    "SceneRecord",
    "SceneTransform",
    "SplitSceneRecord",
    "StorageBackend",
    "manifest_path",
    "read_manifest",
    "write_manifest",
]
