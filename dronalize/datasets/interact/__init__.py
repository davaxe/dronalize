__dronalize_builtin__ = {"datasets": ["interact"]}

from dronalize.datasets import _registry
from dronalize.datasets.interact.loader import InteractionLoader
from dronalize.datasets.interact.map.builder import InteractMapBuilder

__all__ = ["InteractMapBuilder", "InteractionLoader"]

_registry.register(
    _registry.DatasetDescriptor(
        name="interact",
        loader_factory=InteractionLoader,
        default_config=InteractionLoader.default_config(),
        default_map_config=InteractionLoader.default_map_config(),
        has_map=True,
    ).with_all_splits()
)
