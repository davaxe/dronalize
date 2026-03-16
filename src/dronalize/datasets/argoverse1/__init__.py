__dronalize_builtin__ = {"datasets": ["argoverse1"]}

from dronalize.datasets import _registry
from dronalize.datasets.argoverse1 import _scope
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.map.builder import Argoverse1MapBuilder

__all__ = ["Argoverse1Loader", "Argoverse1MapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="argoverse1",
        loader_factory=Argoverse1Loader,
        default_config=Argoverse1Loader.default_config(),
        default_map_config=Argoverse1Loader.default_map_config(),
        has_map=True,
        execution_scope_fn=_scope.argoverse1_execution_scope,
    ).with_all_splits()
)
