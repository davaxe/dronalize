from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.argoverse1 import _scope
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.map.builder import Argoverse1MapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "argoverse1",
    Argoverse1Loader,
    execution_scope_fn=_scope.argoverse1_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "Argoverse1Loader", "Argoverse1MapBuilder"]
