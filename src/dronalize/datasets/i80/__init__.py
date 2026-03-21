from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.i80.loader import I80Loader
from dronalize.datasets.i80.map.builder import I80MapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="i80",
    loader_factory=I80Loader,
    default_config=I80Loader.default_config(),
    default_map_config=I80Loader.default_map_config(),
    has_map=True,
    predefined_splits=list(I80Loader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "I80Loader", "I80MapBuilder"]
