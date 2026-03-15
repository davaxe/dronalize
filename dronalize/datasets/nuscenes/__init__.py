__dronalize_builtin__ = {"datasets": ["nuscenes"]}

from dronalize.datasets import _registry
from dronalize.datasets.nuscenes import _scope
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.builder import NuScenesMapBuilder

__all__ = ["NuScenesLoader", "NuScenesMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="nuscenes",
        loader_factory=NuScenesLoader,
        default_config=NuScenesLoader.default_config(),
        default_map_config=NuScenesLoader.default_map_config(),
        has_map=True,
        predefined_splits=[],
        execution_scope_fn=_scope.nuscenes_execution_scope,
    )
)
