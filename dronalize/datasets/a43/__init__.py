from dronalize.datasets import registry
from dronalize.datasets.a43.graph_builder import A43GraphBuilder
from dronalize.datasets.a43.loader import A43Loader

__all__ = ["A43GraphBuilder", "A43Loader"]

registry.register(
    registry.DatasetDescriptor(
        name="a43",
        loader_factory=A43Loader,
        default_config=A43Loader.default_config(),
        map_mode=registry.MapMode.BUILDER_ONLY,
        predefined_splits=None,
    )
)
