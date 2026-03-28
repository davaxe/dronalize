from dronalize.datasets.argoverse2.loader import Argoverse2Loader
from dronalize.datasets.argoverse2.maps.builder import Argoverse2MapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("argoverse2", Argoverse2Loader, has_map=True)

__all__ = ["DESCRIPTOR", "Argoverse2Loader", "Argoverse2MapBuilder"]
