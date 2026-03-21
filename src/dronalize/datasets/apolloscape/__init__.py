from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader

DESCRIPTOR = DatasetDescriptor(
    name="apolloscape",
    loader_factory=ApolloScapeLoader,
    default_config=ApolloScapeLoader.default_config(),
    default_map_config=ApolloScapeLoader.default_map_config(),
    has_map=True,
    predefined_splits=list(ApolloScapeLoader.predefined_splits()),
)

__all__ = ["DESCRIPTOR", "ApolloScapeLoader"]
