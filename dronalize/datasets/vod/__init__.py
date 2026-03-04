from dronalize.datasets import registry
from dronalize.datasets.vod.graph_builder import VODMapGraphBuilder
from dronalize.datasets.vod.loader import VodLoader

__all__ = ["VODMapGraphBuilder", "VodLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="vod",
        loader_factory=VodLoader,
        default_config=VodLoader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
