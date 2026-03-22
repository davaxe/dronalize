from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.interact.loader import InteractionLoader
from dronalize.datasets.interact.map.builder import InteractMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "interact", InteractionLoader, InteractionLoader, has_map=True
)

__all__ = ["DESCRIPTOR", "InteractMapBuilder", "InteractionLoader"]
