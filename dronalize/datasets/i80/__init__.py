__dronalize_builtin__ = {"datasets": ["i80"]}

from dronalize.datasets import _registry
from dronalize.datasets.i80.loader import I80Loader
from dronalize.datasets.i80.map.builder import I80MapBuilder

__all__ = ["I80Loader", "I80MapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="i80",
        loader_factory=I80Loader,
        default_config=I80Loader.default_config(),
        default_map_config=I80Loader.default_map_config(),
        has_map=True,
        predefined_splits=[],
    )
)
