from dronalize.datasets import registry
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader

__all__ = ["ApolloScapeLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="apolloscape",
        loader_factory=ApolloScapeLoader,
        has_map=False,
        predefined_splits=None,
    )
)
