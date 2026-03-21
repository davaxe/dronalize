from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.nuscenes import _scope
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.builder import NuScenesMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="nuscenes",
    loader_factory=NuScenesLoader,
    default_config=NuScenesLoader.default_config(),
    default_map_config=NuScenesLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(NuScenesLoader.predefined_splits()),
    execution_scope_fn=_scope.nuscenes_execution_scope,
)

__all__ = ["DESCRIPTOR", "NuScenesLoader", "NuScenesMapBuilder"]
