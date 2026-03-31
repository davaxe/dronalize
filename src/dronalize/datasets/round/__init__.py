from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor
from dronalize.datasets.round import scope as _scope
from dronalize.datasets.round.loader import RounDLoader
from dronalize.datasets.round.maps.builder import RounDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "round",
    RounDLoader,
    execution_scope_fn=_scope.round_execution_scope,
    capabilities=DatasetCapabilities.MAP_AVAILABLE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "RounDLoader", "RounDMapBuilder"]
