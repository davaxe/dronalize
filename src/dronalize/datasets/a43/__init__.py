from dronalize.datasets.a43.loader import A43Loader
from dronalize.datasets.a43.maps.builder import A43MapBuilder
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader("a43", A43Loader, infer_capabilities=True)

__all__ = ["DESCRIPTOR", "A43Loader", "A43MapBuilder"]
