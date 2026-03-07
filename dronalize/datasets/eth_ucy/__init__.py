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
        has_map=False,
    ).with_all_splits()
)

registry.register(
    registry.DatasetDescriptor(
        name="hotel",
        loader_factory=HotelLoader,
        has_map=False,
    ).with_all_splits()
)

registry.register(
    registry.DatasetDescriptor(
        name="univ",
        loader_factory=UnivLoader,
        has_map=False,
    ).with_all_splits()
)

registry.register(
    registry.DatasetDescriptor(
        name="zara1",
        loader_factory=Zara1Loader,
        has_map=False,
    ).with_all_splits()
)
