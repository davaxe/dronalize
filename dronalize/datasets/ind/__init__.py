from dronalize.datasets import registry
from dronalize.datasets.ind.graph_builder import InDGraphBuilder
from dronalize.datasets.ind.loader import InDLoader

__all__ = ["InDGraphBuilder", "InDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="ind",
        loader_factory=InDLoader,
        has_map=True,
        predefined_splits=None,
    )
)
