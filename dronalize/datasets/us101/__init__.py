from dronalize.datasets import registry
from dronalize.datasets.us101.graph_builder import US101GraphBuilder
from dronalize.datasets.us101.loader import US101Loader

__all__ = ["US101GraphBuilder", "US101Loader"]

registry.register(
    registry.DatasetDescriptor(
        name="us101",
        loader_factory=US101Loader,
        default_config=US101Loader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
