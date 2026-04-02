from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor
from dronalize.datasets.us101 import scope as _scope
from dronalize.datasets.us101.loader import US101Loader
from dronalize.datasets.us101.maps.builder import US101MapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "us101",
    US101Loader,
    execution_scope_fn=_scope.us101_execution_scope,
    capabilities=DatasetCapabilities.HIGHWAY_PIPELINE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "US101Loader", "US101MapBuilder"]
