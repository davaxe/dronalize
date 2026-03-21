from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.argoverse1 import _scope
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.map.builder import Argoverse1MapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="argoverse1",
    loader_factory=Argoverse1Loader,
    default_config=Argoverse1Loader.default_config(),
    default_map_config=Argoverse1Loader.default_map_config(),
    has_map=True,
    execution_scope_fn=_scope.argoverse1_execution_scope,
    predefined_splits=list(Argoverse1Loader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "Argoverse1Loader", "Argoverse1MapBuilder"]
