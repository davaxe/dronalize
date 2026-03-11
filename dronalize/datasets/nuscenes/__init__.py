from dronalize.datasets import registry
from dronalize.datasets.nuscenes import _lifecycle
from dronalize.datasets.nuscenes.loader import NuScenesLoader
from dronalize.datasets.nuscenes.map.graph_builder import NuScenesMapGraphBuilder

__all__ = ["NuScenesLoader", "NuScenesMapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="nuscenes",
        loader_factory=NuScenesLoader,
        default_config=NuScenesLoader.default_config(),
        default_map_config=NuScenesLoader.default_map_config(),
        map_mode=registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        lifecycle_context=_lifecycle.nuscenes_lifecylce_context,
    )
)
