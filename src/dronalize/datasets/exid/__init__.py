from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.exid import _scope
from dronalize.datasets.exid.loader import ExiDLoader
from dronalize.datasets.exid.map.builder import ExiDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "exid",
    ExiDLoader,
    execution_scope_fn=_scope.exid_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "ExiDLoader", "ExiDMapBuilder"]
