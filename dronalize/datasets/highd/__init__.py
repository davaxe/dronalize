from dronalize.datasets import registry
from dronalize.datasets.highd.graph_builder import HighDMapGraphBuilder
from dronalize.datasets.highd.loader import HighDLoader

__all__ = ["HighDLoader", "HighDMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="highd",
        loader_factory=HighDLoader,
        default_config=HighDLoader.default_config(),
        default_map_config=HighDLoader.default_map_config(),
        map_mode=registry.MapMode.LAZY_KEYED,
        predefined_splits=[],
    )
)
