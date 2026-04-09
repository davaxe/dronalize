from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.round import runtime_context as _runtime_context
from dronalize.datasets.round.loader import RounDLoader as _Loader

DATASET_SPEC = DatasetSpec.from_loader(
    "round",
    _Loader,
    runtime_context_fn=_runtime_context.round_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
