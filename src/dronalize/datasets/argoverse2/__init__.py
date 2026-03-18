from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.argoverse2.loader import Argoverse2Loader
from dronalize.datasets.argoverse2.map.builder import Argoverse2MapBuilder

DESCRIPTOR = DatasetDescriptor(
    name="argoverse2",
    loader_factory=Argoverse2Loader,
    default_config=Argoverse2Loader.default_config(),
    default_map_config=Argoverse2Loader.default_map_config(),
    has_map=True,
    predefined_splits=list(Argoverse2Loader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "Argoverse2Loader", "Argoverse2MapBuilder"]
