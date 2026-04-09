from dronalize.datasets.highd.loader import HighDLoader as _Loader
from dronalize.datasets.registry import DatasetCapabilities, DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "highd",
    _Loader,
    capabilities=DatasetCapabilities.LANE_CHANGE_SAMPLING,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
