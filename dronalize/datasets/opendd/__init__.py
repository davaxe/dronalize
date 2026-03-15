__dronalize_builtin__ = {"datasets": ["opendd"]}

from dronalize.datasets import _registry
from dronalize.datasets.opendd.loader import OpenDDLoader
from dronalize.datasets.opendd.map.builder import OpenDDMapBuilder

__all__ = ["OpenDDLoader", "OpenDDMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="opendd",
        loader_factory=OpenDDLoader,
        default_config=OpenDDLoader.default_config(),
        default_map_config=OpenDDLoader.default_map_config(),
        has_map=True,
        predefined_splits=[],
    )
)
