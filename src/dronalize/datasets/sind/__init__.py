from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.sind.loader import SindLoader

DESCRIPTOR = DatasetDescriptor.from_loader("sind", SindLoader, has_map=True)

__all__ = ["DESCRIPTOR", "SindLoader"]
