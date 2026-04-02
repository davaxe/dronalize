from dronalize.datasets.registry import DatasetDescriptor
from dronalize.datasets.sind.loader import SindLoader

DESCRIPTOR = DatasetDescriptor.from_loader("sind", SindLoader, infer_capabilities=True)

__all__ = ["DESCRIPTOR", "SindLoader"]
