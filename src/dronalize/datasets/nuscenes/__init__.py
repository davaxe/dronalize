from dronalize.datasets.nuscenes import scope as _scope
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.maps.builder import NuScenesMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "nuscenes", NuScenesLoader, execution_scope_fn=_scope.nuscenes_execution_scope, has_map=True
)

__all__ = ["DESCRIPTOR", "NuScenesLoader", "NuScenesMapBuilder"]
