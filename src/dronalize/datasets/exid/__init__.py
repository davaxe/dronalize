from dronalize.datasets.exid import runtime_context as _runtime_context
from dronalize.datasets.exid.loader import ExiDLoader as _Loader
from dronalize.datasets.registry import DatasetCapabilities, DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "exid",
    _Loader,
    runtime_context_fn=_runtime_context.exid_runtime_context,
    capabilities=DatasetCapabilities.LANE_CHANGE_SAMPLING,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
