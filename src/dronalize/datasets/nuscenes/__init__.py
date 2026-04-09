from dronalize.datasets.nuscenes import runtime_context as _runtime_context
from dronalize.datasets.nuscenes.loader import NuScenesLoader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "nuscenes",
    _Loader,
    runtime_context_fn=_runtime_context.nuscenes_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
