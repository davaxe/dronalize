__dronalize_builtin__ = {"datasets": ["ind"]}

from dronalize.datasets import _registry
from dronalize.datasets.ind import _scope
from dronalize.datasets.ind.loader import InDLoader
from dronalize.datasets.ind.map.builder import InDMapBuilder

__all__ = ["InDLoader", "InDMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="ind",
        loader_factory=InDLoader,
        default_config=InDLoader.default_config(),
        default_map_config=InDLoader.default_map_config(),
        map_mode=_registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        execution_scope_fn=_scope.ind_execution_scope,
    )
)
