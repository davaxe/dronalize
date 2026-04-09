from dronalize.datasets.ind import runtime_context as _runtime_context
from dronalize.datasets.ind.loader import InDLoader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "ind",
    _Loader,
    runtime_context_fn=_runtime_context.ind_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
