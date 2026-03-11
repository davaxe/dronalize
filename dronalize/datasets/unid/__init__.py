from dronalize.datasets import registry
from dronalize.datasets.unid import _lifecycle
from dronalize.datasets.unid.graph_builder import UniDGraphBuilder
from dronalize.datasets.unid.loader import UniDLoader

__all__ = ["UniDGraphBuilder", "UniDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="unid",
        loader_factory=UniDLoader,
        default_config=UniDLoader.default_config(),
        default_map_config=UniDLoader.default_map_config(),
        map_mode=registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        lifecycle_context=_lifecycle.unid_lifecycle_context,
    )
)
