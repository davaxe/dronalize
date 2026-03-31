from dronalize.datasets.ad4che.loader import AD4CHELoader
from dronalize.datasets.ad4che.maps.builder import AD4CHEMapBuilder
from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "ad4che",
    AD4CHELoader,
    capabilities=DatasetCapabilities.MAP_AVAILABLE | DatasetCapabilities.HIGHWAY_PIPELINE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "AD4CHELoader", "AD4CHEMapBuilder"]
