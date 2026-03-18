from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.waymo.loader import WaymoLoader
from dronalize.datasets.waymo.map.builder import WaymoMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="waymo",
    loader_factory=WaymoLoader,
    default_config=WaymoLoader.default_config(),
    default_map_config=WaymoLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(WaymoLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "WaymoLoader", "WaymoMapBuilder"]
