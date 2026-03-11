__dronalize_builtin__ = {"datasets": ["vod"]}

from dronalize.datasets import _registry
from dronalize.datasets.vod import _scope
from dronalize.datasets.vod.loader import VodLoader
from dronalize.datasets.vod.map.builder import VODMapBuilder

__all__ = ["VODMapBuilder", "VodLoader"]

_registry.register(
    _registry.DatasetDescriptor(
        name="vod",
        loader_factory=VodLoader,
        default_config=VodLoader.default_config(),
        default_map_config=VodLoader.default_map_config(),
        map_mode=_registry.MapMode.SHARED_SINGLE,
        predefined_splits=[],
        execution_scope_fn=_scope.vod_execution_scope,
    )
)
