from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.highd.loader import HighDLoader
from dronalize.datasets.highd.map.builder import HighDMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="highd",
    loader_factory=HighDLoader,
    default_config=HighDLoader.default_config(),
    default_map_config=HighDLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(HighDLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "HighDLoader", "HighDMapBuilder"]
