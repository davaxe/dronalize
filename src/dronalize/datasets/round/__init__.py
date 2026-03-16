__dronalize_builtin__ = {"datasets": ["round"]}

from dronalize.datasets import _registry
from dronalize.datasets.round import _scope
from dronalize.datasets.round.loader import RounDLoader
from dronalize.datasets.round.map.builder import RounDMapBuilder

__all__ = ["RounDLoader", "RounDMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="round",
        loader_factory=RounDLoader,
        default_config=RounDLoader.default_config(),
        default_map_config=RounDLoader.default_map_config(),
        has_map=True,
        predefined_splits=[],
        execution_scope_fn=_scope.round_execution_scope,
    )
)
