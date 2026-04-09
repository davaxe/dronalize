from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.vod import runtime_context as _runtime_context
from dronalize.datasets.vod.loader import VodLoader as _Loader

DATASET_SPEC = DatasetSpec.from_loader(
    "vod",
    _Loader,
    runtime_context_fn=_runtime_context.vod_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
