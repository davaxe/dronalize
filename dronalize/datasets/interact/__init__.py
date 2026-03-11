from dronalize.datasets import registry
from dronalize.datasets.interact.graph_builder import InteractMapGraphBuilder
from dronalize.datasets.interact.loader import InteractionLoader

__all__ = ["InteractMapGraphBuilder", "InteractionLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="interact",
        loader_factory=InteractionLoader,
        default_config=InteractionLoader.default_config(),
        map_mode=registry.MapMode.BUILDER_ONLY,
    ).with_all_splits()
)
