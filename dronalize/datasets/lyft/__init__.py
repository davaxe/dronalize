from dronalize.config.map import MapConfig
from dronalize.datasets import registry
from dronalize.datasets.lyft.lifecycle import lyft_lifecylce_context
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.graph_builder import LyftMapGraphBuilder

__all__ = ["LyftLoader", "LyftMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="lyft",
        loader_factory=LyftLoader,
        default_config=LyftLoader.default_config(),
        lifecycle_context=lyft_lifecylce_context,
        default_map_config=MapConfig.default(),
        map_mode=registry.MapMode.SHARED_SINGLE,
    ).with_splits("train", "val")
)
