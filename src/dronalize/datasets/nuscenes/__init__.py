from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.nuscenes import _scope
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.builder import NuScenesMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "nuscenes",
    NuScenesLoader,
    execution_scope_fn=_scope.nuscenes_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "NuScenesLoader", "NuScenesMapBuilder"]
