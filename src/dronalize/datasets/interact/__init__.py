from dronalize.datasets.interact.loader import InteractionLoader
from dronalize.datasets.interact.maps.builder import InteractMapBuilder
from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "interact",
    InteractionLoader,
    capabilities=DatasetCapabilities.MAP_AVAILABLE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "InteractMapBuilder", "InteractionLoader"]
