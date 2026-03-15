__dronalize_builtin__ = {"datasets": ["argoverse2"]}

from dronalize.datasets import _registry
from dronalize.datasets.argoverse2.loader import Argoverse2Loader
from dronalize.datasets.argoverse2.map.builder import Argoverse2MapBuilder

__all__ = ["Argoverse2Loader", "Argoverse2MapBuilder"]
_registry.register(
    _registry.DatasetDescriptor(
        name="argoverse2",
        loader_factory=Argoverse2Loader,
        default_config=Argoverse2Loader.default_config(),
        default_map_config=Argoverse2Loader.default_map_config(),
        has_map=True,
    ).with_all_splits()
)
