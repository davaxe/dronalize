from dronalize.datasets import registry
from dronalize.datasets.nuscenes.lifecycle import nuscenes_lifecylce_context
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.graph_builder import NuScenesMapGraphBuilder

__all__ = ["NuScenesLoader", "NuScenesMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="nuscenes",
        loader_factory=NuScenesLoader,
        default_config=NuScenesLoader.default_config(),
        lifecycle_context=nuscenes_lifecylce_context,
        map_mode=registry.MapMode.SHARED_KEYED,
        predefined_splits=None,
    )
)
