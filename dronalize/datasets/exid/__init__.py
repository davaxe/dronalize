from dronalize.datasets import registry
from dronalize.datasets.exid import _lifecycle
from dronalize.datasets.exid.graph_builder import ExiDGraphBuilder
from dronalize.datasets.exid.loader import ExiDLoader

__all__ = ["ExiDGraphBuilder", "ExiDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="exid",
        loader_factory=ExiDLoader,
        default_config=ExiDLoader.default_config(),
        default_map_config=ExiDLoader.default_map_config(),
        map_mode=registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        lifecycle_context=_lifecycle.exid_lifecycle_context,
    )
)
