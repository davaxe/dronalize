from dronalize.datasets.i80.loader import I80Loader
from dronalize.datasets.i80.maps.builder import I80MapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("i80", I80Loader, has_map=True)

__all__ = ["DESCRIPTOR", "I80Loader", "I80MapBuilder"]
