from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.ind import _scope
from dronalize.datasets.ind.loader import InDLoader
from dronalize.datasets.ind.map.builder import InDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "ind",
    InDLoader,
    execution_scope_fn=_scope.ind_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "InDLoader", "InDMapBuilder"]
