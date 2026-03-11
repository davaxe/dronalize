from dronalize.datasets import registry
from dronalize.datasets.sind.loader import SindLoader

__all__ = ["SindLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="sind",
        loader_factory=SindLoader,
        default_config=SindLoader.default_config(),
        map_mode=registry.MapMode.BUILDER_ONLY,
        predefined_splits=None,
    )
)
