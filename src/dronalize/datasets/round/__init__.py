from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.round import _scope
from dronalize.datasets.round.loader import RounDLoader
from dronalize.datasets.round.map.builder import RounDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "round",
    RounDLoader,
    execution_scope_fn=_scope.round_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "RounDLoader", "RounDMapBuilder"]
