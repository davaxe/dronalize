__dronalize_builtin__ = {"datasets": ["a43"]}

from dronalize.datasets import _registry
from dronalize.datasets.a43.loader import A43Loader
from dronalize.datasets.a43.map.builder import A43MapBuilder

__all__ = ["A43Loader", "A43MapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="a43",
        loader_factory=A43Loader,
        default_config=A43Loader.default_config(),
        default_map_config=A43Loader.default_map_config(),
        has_map=True,
        predefined_splits=[],
    )
)
