from dronalize.datasets.ad4che.loader import AD4CHELoader
from dronalize.datasets.ad4che.maps.builder import AD4CHEMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("ad4che", AD4CHELoader, has_map=True)

__all__ = ["DESCRIPTOR", "AD4CHELoader", "AD4CHEMapBuilder"]
