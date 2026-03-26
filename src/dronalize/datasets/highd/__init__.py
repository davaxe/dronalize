from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.highd.loader import HighDLoader
from dronalize.datasets.highd.map.builder import HighDMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("highd", HighDLoader, has_map=True)

__all__ = ["DESCRIPTOR", "HighDLoader", "HighDMapBuilder"]
