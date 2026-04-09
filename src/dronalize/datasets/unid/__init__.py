from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.unid import runtime_context as _runtime_context
from dronalize.datasets.unid.loader import UniDLoader as _Loader

DATASET_SPEC = DatasetSpec.from_loader(
    "unid",
    _Loader,
    runtime_context_fn=_runtime_context.unid_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
