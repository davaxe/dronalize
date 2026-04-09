from dronalize.datasets.lyft import runtime_context as _runtime_context
from dronalize.datasets.lyft.loader import LyftLoader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "lyft",
    _Loader,
    runtime_context_fn=_runtime_context.lyft_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
