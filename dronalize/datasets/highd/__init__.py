from dronalize.datasets import registry
from dronalize.datasets.highd.graph_builder import HighDMapGraphBuilder
from dronalize.datasets.highd.loader import HighDLoader

__all__ = ["HighDLoader", "HighDMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="highd",
        loader_factory=HighDLoader,
        default_config=HighDLoader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
