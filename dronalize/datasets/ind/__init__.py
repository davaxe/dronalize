from dronalize.datasets import registry
from dronalize.datasets.ind import _lifecycle
from dronalize.datasets.ind.graph_builder import InDGraphBuilder
from dronalize.datasets.ind.loader import InDLoader

__all__ = ["InDGraphBuilder", "InDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="ind",
        loader_factory=InDLoader,
        default_config=InDLoader.default_config(),
        default_map_config=InDLoader.default_map_config(),
        map_mode=registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        lifecycle_context=_lifecycle.ind_lifecycle_context,
    )
)
