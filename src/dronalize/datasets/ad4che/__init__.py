from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.ad4che.loader import AD4CHELoader
from dronalize.datasets.ad4che.map.builder import AD4CHEMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="ad4che",
    loader_factory=AD4CHELoader,
    default_config=AD4CHELoader.default_config(),
    default_map_config=AD4CHELoader.default_map_config(),
    has_map=True,
    predefined_splits=list(AD4CHELoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "AD4CHELoader", "AD4CHEMapBuilder"]
