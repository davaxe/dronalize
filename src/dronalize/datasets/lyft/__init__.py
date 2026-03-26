from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.lyft import _scope
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.map.builder import LyftMapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "lyft",
    LyftLoader,
    execution_scope_fn=_scope.lyft_execution_scope,
    has_map=True,
)

__all__ = ["DESCRIPTOR", "LyftLoader", "LyftMapBuilder"]
