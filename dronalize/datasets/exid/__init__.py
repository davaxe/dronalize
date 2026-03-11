__dronalize_builtin__ = {"datasets": ["exid"]}

from dronalize.datasets import _registry
from dronalize.datasets.exid import _scope
from dronalize.datasets.exid.loader import ExiDLoader
from dronalize.datasets.exid.map.builder import ExiDMapBuilder

__all__ = ["ExiDLoader", "ExiDMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="exid",
        loader_factory=ExiDLoader,
        default_config=ExiDLoader.default_config(),
        default_map_config=ExiDLoader.default_map_config(),
        map_mode=_registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        execution_scope_fn=_scope.exid_execution_scope,
    )
)
