from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader

DESCRIPTOR = DatasetDescriptor.from_loader("apolloscape", ApolloScapeLoader, has_map=True)

__all__ = ["DESCRIPTOR", "ApolloScapeLoader"]
