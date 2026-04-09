from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.waymo.loader import WaymoLoader as _Loader

DATASET_SPEC = DatasetSpec.from_loader("waymo", _Loader, infer_capabilities=True)

__all__ = ["DATASET_SPEC"]
