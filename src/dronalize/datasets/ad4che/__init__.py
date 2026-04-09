from dronalize.datasets.ad4che.loader import AD4CHELoader as _Loader
from dronalize.datasets.registry import DatasetCapabilities, DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "ad4che",
    _Loader,
    capabilities=DatasetCapabilities.LANE_CHANGE_SAMPLING,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
