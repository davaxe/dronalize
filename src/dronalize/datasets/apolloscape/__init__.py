from dronalize.datasets.apolloscape.loader import ApolloScapeLoader as _Loader
from dronalize.datasets.registry import DatasetSpec

DATASET_SPEC = DatasetSpec.from_loader("apolloscape", _Loader, infer_capabilities=True)

__all__ = ["DATASET_SPEC"]
