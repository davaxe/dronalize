__dronalize_builtin__ = {"datasets": ["us101"]}

from dronalize.datasets import _registry
from dronalize.datasets.us101 import _scope
from dronalize.datasets.us101.loader import US101Loader
from dronalize.datasets.us101.map.builder import US101MapBuilder

__all__ = ["US101Loader", "US101MapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="us101",
        loader_factory=US101Loader,
        default_config=US101Loader.default_config(),
        default_map_config=US101Loader.default_map_config(),
        has_map=True,
        execution_scope_fn=_scope.us101_execution_scope,
        predefined_splits=[],
    )
)
