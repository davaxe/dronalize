from dronalize.datasets import registry
from dronalize.datasets.vod import _lifecycle
from dronalize.datasets.vod.graph_builder import VODMapGraphBuilder
from dronalize.datasets.vod.loader import VodLoader

__all__ = ["VODMapGraphBuilder", "VodLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="vod",
        loader_factory=VodLoader,
        default_config=VodLoader.default_config(),
        default_map_config=VodLoader.default_map_config(),
        map_mode=registry.MapMode.SHARED_SINGLE,
        predefined_splits=[],
        lifecycle_context=_lifecycle.vod_lifecycle_context,
    )
)
