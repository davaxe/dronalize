from dronalize.datasets import registry
from dronalize.datasets.ad4che.graph_builder import AD4CHEGraphBuilder
from dronalize.datasets.ad4che.loader import AD4CHELoader

__all__ = ["AD4CHEGraphBuilder", "AD4CHELoader"]

registry.register(
    registry.DatasetDescriptor(
        name="ad4che",
        loader_factory=AD4CHELoader,
        default_config=AD4CHELoader.default_config(),
        has_map=True,
        predefined_splits=None,
    )
)
