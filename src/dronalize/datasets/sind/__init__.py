from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.sind.loader import SindLoader

DESCRIPTOR = DatasetDescriptor(
    name="sind",
    loader_factory=SindLoader,
    default_config=SindLoader.default_config(),
    default_map_config=SindLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(SindLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "SindLoader"]
