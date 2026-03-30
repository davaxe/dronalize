from dronalize.datasets.exid import scope as _scope
from dronalize.datasets.exid.loader import ExiDLoader
from dronalize.datasets.exid.maps.builder import ExiDMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "exid", ExiDLoader, execution_scope_fn=_scope.exid_execution_scope, has_map=True
)

__all__ = ["DESCRIPTOR", "ExiDLoader", "ExiDMapBuilder"]
