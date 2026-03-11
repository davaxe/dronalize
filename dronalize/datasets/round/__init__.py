from dronalize.datasets import registry
from dronalize.datasets.round import _lifecycle
from dronalize.datasets.round.graph_builder import RounDGraphBuilder
from dronalize.datasets.round.loader import RounDLoader

__all__ = ["RounDGraphBuilder", "RounDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="round",
        loader_factory=RounDLoader,
        default_config=RounDLoader.default_config(),
        default_map_config=RounDLoader.default_map_config(),
        map_mode=registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        lifecycle_context=_lifecycle.round_lifecylce_context,
    )
)
