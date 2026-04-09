from dronalize.datasets.interact.loader import InteractionLoader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader(
    "interact",
    _Loader,
    infer_capabilities=True,
)

__all__ = ["DATASET_SPEC"]
