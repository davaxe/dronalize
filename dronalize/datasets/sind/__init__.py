from dronalize.datasets import registry
from dronalize.datasets.sind.loader import SindLoader

__all__ = ["SindLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="sind",
        loader_factory=SindLoader,
        has_map=False,
        predefined_splits=None,
    )
)
