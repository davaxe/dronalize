from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.a43.loader import A43Loader
from dronalize.datasets.a43.map.builder import A43MapBuilder

DESCRIPTOR = DatasetDescriptor.from_loader("a43", A43Loader, has_map=True)

__all__ = ["DESCRIPTOR", "A43Loader", "A43MapBuilder"]
