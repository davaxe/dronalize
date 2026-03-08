from dronalize.datasets import registry
from dronalize.datasets.exid.graph_builder import ExiDGraphBuilder
from dronalize.datasets.exid.loader import ExiDLoader

__all__ = ["ExiDGraphBuilder", "ExiDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="exid",
        loader_factory=ExiDLoader,
        has_map=True,
        predefined_splits=None,
    )
)
