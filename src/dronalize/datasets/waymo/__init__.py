__dronalize_builtin__ = {
    "datasets": ["waymo"],
    "optional_dependencies": ["google.protobuf"],
    "extra": "waymo",
}

from dronalize.datasets import _registry

_registry.ensure_builtin_dependencies(__name__, __dronalize_builtin__)

from dronalize.datasets.waymo.loader import WaymoLoader  # noqa: E402
from dronalize.datasets.waymo.map.builder import WaymoMapBuilder  # noqa: E402

__all__ = ["WaymoLoader", "WaymoMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="waymo",
        loader_factory=WaymoLoader,
        default_config=WaymoLoader.default_config(),
        default_map_config=WaymoLoader.default_map_config(),
        has_map=True,
    ).with_all_splits()
)
