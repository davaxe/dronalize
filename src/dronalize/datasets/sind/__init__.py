from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.sind.loader import SindLoader as _Loader

DATASET_SPEC = DatasetSpec.from_loader("sind", _Loader, infer_capabilities=True)

__all__ = ["DATASET_SPEC"]
