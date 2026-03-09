from dronalize.datasets import registry
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader

__all__ = ["ApolloScapeLoader"]

registry.register(
    registry.DatasetDescriptor(
        name="apolloscape",
        loader_factory=ApolloScapeLoader,
        default_config=ApolloScapeLoader.default_config(),
        has_map=False,
    ).with_splits("train", "val")
)
