from dronalize.datasets.lyft import scope as _scope
from dronalize.datasets.lyft.loader import LyftLoader
from dronalize.datasets.lyft.maps.builder import LyftMapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "lyft",
    LyftLoader,
    execution_scope_fn=_scope.lyft_execution_scope,
    infer_capabilities=True,
)

__all__ = ["DESCRIPTOR", "LyftLoader", "LyftMapBuilder"]
