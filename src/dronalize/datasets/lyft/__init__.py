from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.lyft import _scope
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.builder import LyftMapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="lyft",
    loader_factory=LyftLoader,
    default_config=LyftLoader.default_config(),
    default_map_config=LyftLoader.default_map_config(),
    execution_scope_fn=_scope.lyft_execution_scope,
    has_map=True,
    predefined_splits=list(LyftLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "LyftLoader", "LyftMapBuilder"]
