"""Persisted storage contracts and export-facing configuration.

## Import guide

```python
from prejectory.io import (
    DatasetManifest,
    RecordTransform,
    SceneRecord,
    SceneTransform,
    SplitSceneRecord,
    read_manifest,
    write_manifest,
)
```

Output configuration models live under [`prejectory.config.models`][] so the
runtime and CLI share one canonical configuration surface.

## Related modules

- [`prejectory.io.adapters`][] for optional Torch and PyG adapter layers
"""

from prejectory.io.base import RecordTransform, SceneTransform, StorageBackend
from prejectory.io.manifest import (
    DatasetManifest,
    PredictionTaskManifest,
    manifest_path,
    read_manifest,
    write_manifest,
)
from prejectory.io.records import PredictionBounds, SceneRecord, SplitSceneRecord

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
