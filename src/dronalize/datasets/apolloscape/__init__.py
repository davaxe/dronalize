from dronalize.datasets.apolloscape.loader import ApolloScapeLoader
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTOR = DatasetDescriptor.from_loader(
    "apolloscape", ApolloScapeLoader, infer_capabilities=True
)

__all__ = ["DESCRIPTOR", "ApolloScapeLoader"]
