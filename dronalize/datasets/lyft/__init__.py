from dronalize.core.datatypes.map_config import MapConfig
from dronalize.datasets import registry
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.graph_builder import LyftMapGraphBuilder
from dronalize.datasets.lyft.registry import lyft_lifecylce_context

__all__ = ["LyftLoader", "LyftMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="lyft",
        loader_factory=LyftLoader,
        default_config=LyftLoader.default_config(),
        lifecycle_context=lyft_lifecylce_context,
        default_map_config=MapConfig.default(),
        has_map=True,
        predefined_splits=None,
    )
)
