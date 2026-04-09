from dronalize.datasets.argoverse1 import runtime_context as _runtime_context
from dronalize.datasets.argoverse1.loader import Argoverse1Loader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "argoverse1",
    _Loader,
    runtime_context_fn=_runtime_context.argoverse1_runtime_context,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
