from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.ad4che.loader import AD4CHELoader
from dronalize.datasets.ad4che.map.builder import AD4CHEMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("ad4che", AD4CHELoader, AD4CHELoader, has_map=True)

__all__ = ["DESCRIPTOR", "AD4CHELoader", "AD4CHEMapBuilder"]
