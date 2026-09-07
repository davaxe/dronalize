"""Optional framework adapters built on top of persisted dataset readers.

## Import guide

``python
from prejectory.io.adapters import HeteroSceneDataset, TorchSceneDataset
``

This package groups higher-level dataset adapters for downstream ML code. The
exports are loaded lazily so optional dependencies such as Torch or
Torch-Geometric are only imported when their adapters are actually requested.

Use [`prejectory.io.readers`][] when you want framework-neutral records. Use this
package when you want those records exposed through Torch or PyG dataset
surfaces.

## Related modules

- [`prejectory.io.readers`][] for framework-neutral persisted dataset readers
- [`prejectory.io`][] for storage contracts and export configuration
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from prejectory.core.optional import lazy_dir, resolve_lazy_export

if TYPE_CHECKING:
    from prejectory.io.adapters.pyg import (
        HeteroForecastDataset,
        HeteroSceneDataset,
        IterableHeteroForecastDataset,
        IterableHeteroSceneDataset,
        collate_forecast_hetero_with_time_padding,
        collate_hetero_with_time_padding,
    )
    from prejectory.io.adapters.torch import (
        IterableTorchForecastDataset,
        IterableTorchSceneDataset,
        TorchForecastDataset,
        TorchForecastRecord,
        TorchSceneDataset,
        TorchSceneRecord,
        to_torch_scene_record,
    )

__all__ = [
    "HeteroForecastDataset",
    "HeteroSceneDataset",
    "IterableHeteroForecastDataset",
    "IterableHeteroSceneDataset",
    "IterableTorchForecastDataset",
    "IterableTorchSceneDataset",
    "TorchForecastDataset",
    "TorchForecastRecord",
    "TorchSceneDataset",
    "TorchSceneRecord",
    "collate_forecast_hetero_with_time_padding",
    "collate_hetero_with_time_padding",
    "to_torch_scene_record",
]

__lazy_exports__: dict[str, tuple[str, str]] = {
    name: (
        "prejectory.io.adapters.torch"
        if "Torch" in name or name == "to_torch_scene_record"
        else "prejectory.io.adapters.pyg",
        name,
    )
    for name in __all__
}


def __getattr__(name: str) -> object:
    """Resolve optional adapter exports lazily."""
    return resolve_lazy_export(globals(), __lazy_exports__, module_name=__name__, name=name)


def __dir__() -> list[str]:
    """Expose lazy adapter exports during interactive discovery."""
    return lazy_dir(globals(), exported_names=list(__lazy_exports__))
