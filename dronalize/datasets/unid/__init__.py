__dronalize_builtin__ = {"datasets": ["unid"]}

from dronalize.datasets import _registry
from dronalize.datasets.unid import _scope
from dronalize.datasets.unid.loader import UniDLoader
from dronalize.datasets.unid.map.builder import UniDMapBuilder

__all__ = ["UniDLoader", "UniDMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="unid",
        loader_factory=UniDLoader,
        default_config=UniDLoader.default_config(),
        default_map_config=UniDLoader.default_map_config(),
        map_mode=_registry.MapMode.SHARED_KEYED,
        predefined_splits=[],
        execution_scope_fn=_scope.unid_execution_scope,
    )
)
