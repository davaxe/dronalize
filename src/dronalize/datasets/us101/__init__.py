from dronalize.datasets.registry import DatasetCapabilities, DatasetSpec
from dronalize.datasets.us101 import runtime_context as _runtime_context
from dronalize.datasets.us101.loader import US101Loader as _Loader

DATASET_SPEC = DatasetSpec.from_loader(
    "us101",
    _Loader,
    runtime_context_fn=_runtime_context.us101_runtime_context,
    capabilities=DatasetCapabilities.LANE_CHANGE_SAMPLING,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
