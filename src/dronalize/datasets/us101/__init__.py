from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.us101 import _scope
from dronalize.datasets.us101.loader import US101Loader
from dronalize.datasets.us101.map.builder import US101MapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="us101",
    loader_factory=US101Loader,
    default_config=US101Loader.default_config(),
    default_map_config=US101Loader.default_map_config(),
    has_map=True,
    execution_scope_fn=_scope.us101_execution_scope,
    predefined_splits=list(US101Loader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "US101Loader", "US101MapBuilder"]
