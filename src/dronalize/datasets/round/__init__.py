from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.round import _scope
from dronalize.datasets.round.loader import RounDLoader
from dronalize.datasets.round.map.builder import RounDMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="round",
    loader_factory=RounDLoader,
    default_config=RounDLoader.default_config(),
    default_map_config=RounDLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(RounDLoader.predefined_splits()),
    execution_scope_fn=_scope.round_execution_scope,
)

__all__ = ["DESCRIPTOR", "RounDLoader", "RounDMapBuilder"]
