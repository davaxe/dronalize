from dronalize.datasets import registry
from dronalize.datasets.round.graph_builder import RounDGraphBuilder
from dronalize.datasets.round.loader import RounDLoader

__all__ = ["RounDGraphBuilder", "RounDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="round",
        loader_factory=RounDLoader,
        default_config=RounDLoader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
