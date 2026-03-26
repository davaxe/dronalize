from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.unid import _scope
from dronalize.datasets.unid.loader import UniDLoader
from dronalize.datasets.unid.map.builder import UniDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "unid",
    UniDLoader,
    execution_scope_fn=_scope.unid_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "UniDLoader", "UniDMapBuilder"]
