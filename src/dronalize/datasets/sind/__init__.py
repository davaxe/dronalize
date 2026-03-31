from dronalize.datasets.registry import DatasetCapabilities, DatasetDescriptor
from dronalize.datasets.sind.loader import SindLoader

DESCRIPTOR = DatasetDescriptor.from_loader(
    "sind", SindLoader, capabilities=DatasetCapabilities.MAP_AVAILABLE, infer_capabilities=True
)

__all__ = ["DESCRIPTOR", "SindLoader"]
