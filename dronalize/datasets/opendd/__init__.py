from dronalize.datasets import registry
from dronalize.datasets.opendd.graph_builder import OpenDDMapGraphBuilder
from dronalize.datasets.opendd.loader import OpenDDLoader

__all__ = ["OpenDDLoader", "OpenDDMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="opendd",
        loader_factory=OpenDDLoader,
        default_config=OpenDDLoader.default_config(),
        map_mode=registry.MapMode.BUILDER_ONLY,
        predefined_splits=None,
    )
)
