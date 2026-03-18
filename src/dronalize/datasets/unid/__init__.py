from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.unid import _scope
from dronalize.datasets.unid.loader import UniDLoader
from dronalize.datasets.unid.map.builder import UniDMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="unid",
    loader_factory=UniDLoader,
    default_config=UniDLoader.default_config(),
    default_map_config=UniDLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(UniDLoader.predefined_splits()),
    execution_scope_fn=_scope.unid_execution_scope,
)

__all__ = ["DESCRIPTOR", "UniDLoader", "UniDMapBuilder"]
