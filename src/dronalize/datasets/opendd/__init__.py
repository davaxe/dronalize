from dronalize.datasets.opendd.loader import OpenDDLoader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader("opendd", _Loader, infer_capabilities=True)

__all__ = ["DATASET_SPEC"]
