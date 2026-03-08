from dronalize.datasets import registry
from dronalize.datasets.waymo.loader import WaymoLoader
from dronalize.datasets.waymo.map.graph_builder import WaymoMapGraphBuilder

__all__ = ["WaymoLoader", "WaymoMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="waymo",
        loader_factory=WaymoLoader,
        has_map=True,
    ).with_all_splits()
)
