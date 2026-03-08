from dronalize.datasets import registry
from dronalize.datasets.interact.graph_builder import InteractMapGraphBuilder
from dronalize.datasets.interact.loader import InteractionLoader

__all__ = ["InteractMapGraphBuilder", "InteractionLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="interact",
        loader_factory=InteractionLoader,
        has_map=True,
    ).with_all_splits()
)
