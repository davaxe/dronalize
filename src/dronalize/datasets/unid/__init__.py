from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor
from dronalize.datasets.unid import scope as _scope
from dronalize.datasets.unid.loader import UniDLoader
from dronalize.datasets.unid.maps.builder import UniDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "unid",
    UniDLoader,
    execution_scope_fn=_scope.unid_execution_scope,
    capabilities=DatasetCapabilities.MAP_AVAILABLE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "UniDLoader", "UniDMapBuilder"]
