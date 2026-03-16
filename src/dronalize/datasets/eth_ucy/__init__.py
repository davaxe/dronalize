__dronalize_builtin__ = {"datasets": ["eth", "hotel", "univ", "zara1", "zara2"]}

from dronalize.datasets import _registry
from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)

__all__ = ["EthLoader", "HotelLoader", "UnivLoader", "Zara1Loader", "Zara2Loader"]

_registry.register(
    _registry.DatasetDescriptor(
        name="eth",
        loader_factory=EthLoader,
        default_config=EthLoader.default_config(),
        default_map_config=EthLoader.default_map_config(),
        has_map=False,
    ).with_all_splits()
)

_registry.register(
    _registry.DatasetDescriptor(
        name="hotel",
        loader_factory=HotelLoader,
        default_config=HotelLoader.default_config(),
        default_map_config=HotelLoader.default_map_config(),
        has_map=False,
    ).with_all_splits()
)

_registry.register(
    _registry.DatasetDescriptor(
        name="univ",
        loader_factory=UnivLoader,
        default_config=UnivLoader.default_config(),
        default_map_config=UnivLoader.default_map_config(),
        has_map=False,
    ).with_all_splits()
)

_registry.register(
    _registry.DatasetDescriptor(
        name="zara1",
        loader_factory=Zara1Loader,
        default_config=Zara1Loader.default_config(),
        default_map_config=Zara1Loader.default_map_config(),
        has_map=False,
    ).with_all_splits()
)

_registry.register(
    _registry.DatasetDescriptor(
        name="zara2",
        loader_factory=Zara2Loader,
        default_config=Zara2Loader.default_config(),
        default_map_config=Zara2Loader.default_map_config(),
        has_map=False,
    ).with_all_splits()
)
