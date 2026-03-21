from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.opendd.loader import OpenDDLoader
from dronalize.datasets.opendd.map.builder import OpenDDMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="opendd",
    loader_factory=OpenDDLoader,
    default_config=OpenDDLoader.default_config(),
    default_map_config=OpenDDLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(OpenDDLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "OpenDDLoader", "OpenDDMapBuilder"]
