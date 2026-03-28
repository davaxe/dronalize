from dronalize.datasets.interact.loader import InteractionLoader
from dronalize.datasets.interact.maps.builder import InteractMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("interact", InteractionLoader, has_map=True)

__all__ = ["DESCRIPTOR", "InteractMapBuilder", "InteractionLoader"]
