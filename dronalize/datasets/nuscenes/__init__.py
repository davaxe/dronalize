from dronalize.datasets import registry
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.graph_builder import NuScenesMapGraphBuilder

__all__ = ["NuScenesLoader", "NuScenesMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="nuscenes",
        loader_factory=NuScenesLoader,
        has_map=True,
        predefined_splits=None,
    )
)
