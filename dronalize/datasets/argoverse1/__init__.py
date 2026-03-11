from dronalize.datasets import registry
from dronalize.datasets.argoverse1 import _lifecycle
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.map.graph_builder import Argoverse1MapGraphBuilder

__all__ = ["Argoverse1Loader", "Argoverse1MapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="argoverse1",
        loader_factory=Argoverse1Loader,
        default_config=Argoverse1Loader.default_config(),
        default_map_config=Argoverse1Loader.default_map_config(),
        map_mode=registry.MapMode.LAZY_KEYED,
        lifecycle_context=_lifecycle.argoverse1_lifecycle_context,
    ).with_all_splits()
)
