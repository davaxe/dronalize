from dronalize.datasets import registry
from dronalize.datasets.argoverse2.loader import Argoverse2Loader
from dronalize.datasets.argoverse2.map.graph_builder import Argoverse2GraphBuilder

__all__ = ["Argoverse2GraphBuilder", "Argoverse2Loader"]
registry.register(
    registry.DatasetDescriptor(
        name="argoverse2",
        loader_factory=Argoverse2Loader,
        default_config=Argoverse2Loader.default_config(),
        default_map_config=Argoverse2Loader.default_map_config(),
        map_mode=registry.MapMode.LAZY_KEYED,
    ).with_all_splits()
)
