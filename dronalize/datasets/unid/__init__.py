from dronalize.datasets import registry
from dronalize.datasets.unid.graph_builder import UniDGraphBuilder
from dronalize.datasets.unid.loader import UniDLoader

__all__ = ["UniDGraphBuilder", "UniDLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="unid",
        loader_factory=UniDLoader,
        default_config=UniDLoader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
