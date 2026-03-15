__dronalize_builtin__ = {"datasets": ["highd"]}

from dronalize.datasets import _registry
from dronalize.datasets.highd.loader import HighDLoader
from dronalize.datasets.highd.map.builder import HighDMapBuilder

__all__ = ["HighDLoader", "HighDMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="highd",
        loader_factory=HighDLoader,
        default_config=HighDLoader.default_config(),
        default_map_config=HighDLoader.default_map_config(),
        has_map=True,
        predefined_splits=[],
    )
)
