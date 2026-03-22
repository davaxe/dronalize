from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.argoverse2.loader import Argoverse2Loader
from dronalize.datasets.argoverse2.map.builder import Argoverse2MapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader(
    "argoverse2", Argoverse2Loader, Argoverse2Loader, has_map=True
)

__all__ = ["DESCRIPTOR", "Argoverse2Loader", "Argoverse2MapBuilder"]
