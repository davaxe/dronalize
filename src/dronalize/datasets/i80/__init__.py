from dronalize.datasets.i80.loader import I80Loader
from dronalize.datasets.i80.maps.builder import I80MapBuilder
from dronalize.datasets.i80.scope import i80_execution_scope
from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "i80",
    I80Loader,
    capabilities=DatasetCapabilities.HIGHWAY_PIPELINE,
    infer_capabilities=True,
    execution_scope_fn=i80_execution_scope,
)

__all__ = ["DESCRIPTOR", "I80Loader", "I80MapBuilder"]
