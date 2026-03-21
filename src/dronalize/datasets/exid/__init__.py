from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.exid import _scope
from dronalize.datasets.exid.loader import ExiDLoader
from dronalize.datasets.exid.map.builder import ExiDMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="exid",
    loader_factory=ExiDLoader,
    default_config=ExiDLoader.default_config(),
    default_map_config=ExiDLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(ExiDLoader.predefined_splits()),
    execution_scope_fn=_scope.exid_execution_scope,
)

__all__ = ["DESCRIPTOR", "ExiDLoader", "ExiDMapBuilder"]
