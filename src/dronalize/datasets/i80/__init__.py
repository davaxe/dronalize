from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.i80.loader import I80Loader
from dronalize.datasets.i80.map.builder import I80MapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("i80", I80Loader, I80Loader, has_map=True)

__all__ = ["DESCRIPTOR", "I80Loader", "I80MapBuilder"]
