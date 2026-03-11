__dronalize_builtin__ = {"datasets": ["sind"]}

from dronalize.datasets import _registry
from dronalize.datasets.sind.loader import SindLoader

__all__ = ["SindLoader"]

_registry.register(
    _registry.DatasetDescriptor(
        name="sind",
        loader_factory=SindLoader,
        default_config=SindLoader.default_config(),
        default_map_config=SindLoader.default_map_config(),
        map_mode=_registry.MapMode.BUILDER_ONLY,
        predefined_splits=[],
    )
)
