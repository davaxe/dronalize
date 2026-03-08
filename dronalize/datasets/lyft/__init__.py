from dronalize.datasets import registry
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.graph_builder import LyftMapGraphBuilder

__all__ = ["LyftLoader", "LyftMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="lyft",
        loader_factory=LyftLoader,
        has_map=True,
        predefined_splits=None,
    )
)
