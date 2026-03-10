from dronalize.datasets import registry
from dronalize.datasets.argoverse1.loader import Argoverse1Loader
from dronalize.datasets.argoverse1.map.graph_builder import Argoverse1MapGraphBuilder

__all__ = ["Argoverse1Loader", "Argoverse1MapGraphBuilder"]

registry.register(
    registry.DatasetDescriptor(
        name="argoverse1",
        loader_factory=Argoverse1Loader,
        default_config=Argoverse1Loader.default_config(),
        map_mode=registry.MapMode.BUILDER_ONLY,
    ).with_all_splits()
)
