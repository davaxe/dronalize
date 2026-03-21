from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.interact.loader import InteractionLoader
from dronalize.datasets.interact.map.builder import InteractMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="interact",
    loader_factory=InteractionLoader,
    default_config=InteractionLoader.default_config(),
    default_map_config=InteractionLoader.default_map_config(),
    has_map=False,
    predefined_splits=list(InteractionLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "InteractMapBuilder", "InteractionLoader"]
