from dronalize.datasets.i80.loader import I80Loader as _Loader
from dronalize.datasets.i80.runtime_context import i80_runtime_context
from dronalize.datasets.registry import DatasetCapabilities, DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "i80",
    _Loader,
    capabilities=DatasetCapabilities.LANE_CHANGE_SAMPLING,
    infer_capabilities=True,
    runtime_context_fn=i80_runtime_context,
)

__all__ = ["DATASET_SPEC"]
