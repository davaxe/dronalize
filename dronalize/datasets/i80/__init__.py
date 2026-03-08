from dronalize.datasets import registry
from dronalize.datasets.i80.graph_builder import I80GraphBuilder
from dronalize.datasets.i80.loader import I80Loader

__all__ = ["I80GraphBuilder", "I80Loader"]

registry.register(
    registry.DatasetDescriptor(
        name="i80",
        loader_factory=I80Loader,
        has_map=True,
        predefined_splits=None,
    )
)
