from dronalize.datasets import registry
from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)

__all__ = ["EthLoader", "HotelLoader", "UnivLoader", "Zara1Loader", "Zara2Loader"]

registry.register(
    registry.DatasetDescriptor(
        name="eth",
        loader_factory=EthLoader,
        default_config=EthLoader.default_config(),
        map_mode=registry.MapMode.NONE,
    ).with_all_splits()
)

registry.register(
    registry.DatasetDescriptor(
        name="hotel",
        loader_factory=HotelLoader,
        default_config=HotelLoader.default_config(),
        map_mode=registry.MapMode.NONE,
    ).with_all_splits()
)

registry.register(
    registry.DatasetDescriptor(
        name="univ",
        loader_factory=UnivLoader,
        default_config=UnivLoader.default_config(),
        map_mode=registry.MapMode.NONE,
    ).with_all_splits()
)

registry.register(
    registry.DatasetDescriptor(
        name="zara1",
        loader_factory=Zara1Loader,
        default_config=Zara1Loader.default_config(),
        map_mode=registry.MapMode.NONE,
    ).with_all_splits()
)
