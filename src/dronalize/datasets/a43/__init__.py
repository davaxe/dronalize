from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.a43.loader import A43Loader
from dronalize.datasets.a43.map.builder import A43MapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="a43",
    loader_factory=A43Loader,
    default_config=A43Loader.default_config(),
    default_map_config=A43Loader.default_map_config(),
    has_map=True,
    predefined_splits=list(A43Loader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "A43Loader", "A43MapBuilder"]
