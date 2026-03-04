from dronalize.datasets import registry
from dronalize.datasets.opendd.graph_builder import OpenDDMapGraphBuilder
from dronalize.datasets.opendd.loader import OpenDDLoader

__all__ = ["OpenDDLoader", "OpenDDMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="opendd",
        loader_factory=OpenDDLoader,
        default_config=OpenDDLoader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
