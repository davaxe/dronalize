__dronalize_builtin__ = {
    "datasets": ["lyft"],
    "optional_dependencies": ["zarr", "numcodecs", "google.protobuf"],
    "extra": "lyft",
}

from dronalize.datasets import _registry

_registry.ensure_builtin_dependencies(__name__, __dronalize_builtin__)

from dronalize.datasets.lyft import _scope  # noqa: E402
from dronalize.datasets.lyft.loader import LyftLoader  # noqa: E402
from dronalize.datasets.lyft.map.builder import LyftMapBuilder  # noqa: E402

__all__ = ["LyftLoader", "LyftMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="lyft",
        loader_factory=LyftLoader,
        default_config=LyftLoader.default_config(),
        default_map_config=LyftLoader.default_map_config(),
        execution_scope_fn=_scope.lyft_execution_scope,
        map_mode=_registry.MapMode.SHARED_SINGLE,
    ).with_splits("train", "val")
)
