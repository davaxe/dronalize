from dronalize.datasets.highd.loader import HighDLoader
from dronalize.datasets.highd.maps.builder import HighDMapBuilder
from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "highd",
    HighDLoader,
    capabilities=DatasetCapabilities.HIGHWAY_PIPELINE,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "HighDLoader", "HighDMapBuilder"]
