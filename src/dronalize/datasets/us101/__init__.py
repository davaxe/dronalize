from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.us101 import _scope
from dronalize.datasets.us101.loader import US101Loader
from dronalize.datasets.us101.map.builder import US101MapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "us101",
    US101Loader,
    execution_scope_fn=_scope.us101_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "US101Loader", "US101MapBuilder"]
