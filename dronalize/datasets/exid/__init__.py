from dronalize.datasets import registry
from dronalize.datasets.exid.graph_builder import ExiDGraphBuilder
from dronalize.datasets.exid.loader import ExiDLoader

__all__ = ["ExiDGraphBuilder", "ExiDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="exid",
        loader_factory=ExiDLoader,
        default_config=ExiDLoader.default_config(),
        map_mode=registry.MapMode.BUILDER_ONLY,
        predefined_splits=None,
    )
)
