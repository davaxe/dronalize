from dronalize.datasets.argoverse2.loader import Argoverse2Loader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "argoverse2",
    _Loader,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
