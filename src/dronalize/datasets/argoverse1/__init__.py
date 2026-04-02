from dronalize.datasets.argoverse1 import scope as _scope
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.maps.builder import Argoverse1MapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "argoverse1",
    Argoverse1Loader,
    execution_scope_fn=_scope.argoverse1_execution_scope,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "Argoverse1Loader", "Argoverse1MapBuilder"]
