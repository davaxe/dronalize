__dronalize_builtin__ = {"datasets": ["apolloscape"]}

from dronalize.datasets import _registry
from dronalize.datasets.apolloscape.loader import ApolloScapeLoader

__all__ = ["ApolloScapeLoader"]

_registry.register(
    _registry.DatasetDescriptor(
        name="apolloscape",
        loader_factory=ApolloScapeLoader,
        default_config=ApolloScapeLoader.default_config(),
        default_map_config=ApolloScapeLoader.default_map_config(),
        map_mode=_registry.MapMode.NONE,
    ).with_splits("train", "val")
)
