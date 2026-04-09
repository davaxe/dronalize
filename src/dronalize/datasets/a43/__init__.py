from dronalize.datasets.a43.loader import A43Loader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader("a43", _Loader, infer_capabilities=True)

__all__ = ["DATASET_SPEC"]
