from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.ind import _scope
from dronalize.datasets.ind.loader import InDLoader
from dronalize.datasets.ind.map.builder import InDMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="ind",
    loader_factory=InDLoader,
    default_config=InDLoader.default_config(),
    default_map_config=InDLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(InDLoader.predefined_splits()),
    execution_scope_fn=_scope.ind_execution_scope,
)

__all__ = ["DESCRIPTOR", "InDLoader", "InDMapBuilder"]
