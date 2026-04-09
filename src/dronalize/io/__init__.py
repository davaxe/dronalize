"""Persisted storage contracts and export-facing configuration.

## Import guide

```python
from dronalize.io import ExportConfig, DatasetManifest, DatasetWriter
```

This package groups the stable storage-side API:

- [`ExportConfig`][dronalize.io.ExportConfig] and
  [`MDSBackendConfig`][dronalize.io.MDSBackendConfig] for persisted output
  settings
- [`DatasetManifest`][dronalize.io.DatasetManifest] and manifest helpers for
  dataset metadata on disk
- [`DatasetWriter`][dronalize.io.DatasetWriter] for the writer protocol shared
  by storage backends

Reader and adapter layers live in sibling subpackages so optional dependencies
remain lazy and clearly scoped.

## Related modules

- [`dronalize.io.readers`][] for framework-neutral persisted dataset readers
- [`dronalize.io.adapters`][] for optional Torch and PyG adapter layers
"""

from dronalize.io.backends.base import DatasetWriter
from dronalize.io.config import ExportConfig, MDSBackendConfig
from dronalize.io.manifest import (
    DatasetManifest,
    manifest_path,
    read_manifest,
    write_manifest,
)

__all__ = [
    "DatasetManifest",
    "DatasetWriter",
    "ExportConfig",
    "MDSBackendConfig",
    "manifest_path",
    "read_manifest",
    "write_manifest",
]
