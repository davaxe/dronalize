from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor
from dronalize.datasets.waymo.loader import WaymoLoader
from dronalize.datasets.waymo.maps.builder import WaymoMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "waymo", WaymoLoader, capabilities=DatasetCapabilities.MAP_AVAILABLE, infer_capabilities=True
)

__all__ = ["DESCRIPTOR", "WaymoLoader", "WaymoMapBuilder"]
