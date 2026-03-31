from dronalize.datasets.ind import scope as _scope
from dronalize.datasets.ind.loader import InDLoader
from dronalize.datasets.ind.maps.builder import InDMapBuilder
from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "ind",
    InDLoader,
    execution_scope_fn=_scope.ind_execution_scope,
    capabilities=DatasetCapabilities.MAP_AVAILABLE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "InDLoader", "InDMapBuilder"]
