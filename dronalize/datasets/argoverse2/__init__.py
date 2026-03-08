from dronalize.datasets import registry
from dronalize.datasets.argoverse2.loader import Argoverse2Loader
from dronalize.datasets.argoverse2.map.graph_builder import Argoverse2GraphBuilder

__all__ = ["Argoverse2GraphBuilder", "Argoverse2Loader"]
registry.register(
    registry.DatasetDescriptor(
        name="argoverse1",
        loader_factory=Argoverse2Loader,
        has_map=True,
    ).with_all_splits()
)
