from dronalize.datasets.highd.loader import HighDLoader
from dronalize.datasets.highd.maps.builder import HighDMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("highd", HighDLoader, has_map=True)

__all__ = ["DESCRIPTOR", "HighDLoader", "HighDMapBuilder"]
