from dronalize.datasets import registry
from dronalize.datasets.lyft import _lifecycle
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.graph_builder import LyftMapGraphBuilder

__all__ = ["LyftLoader", "LyftMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="lyft",
        loader_factory=LyftLoader,
        default_config=LyftLoader.default_config(),
        default_map_config=LyftLoader.default_map_config(),
        lifecycle_context=_lifecycle.lyft_lifecylce_context,
        map_mode=registry.MapMode.SHARED_SINGLE,
    ).with_splits("train", "val")
)
