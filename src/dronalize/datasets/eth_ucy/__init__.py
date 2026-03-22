from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)

DESCRIPTORS = {
    "eth": DatasetDescriptor.from_loader("eth", EthLoader, EthLoader),
    "hotel": DatasetDescriptor.from_loader("hotel", HotelLoader, HotelLoader),
    "univ": DatasetDescriptor.from_loader("univ", UnivLoader, UnivLoader),
    "zara1": DatasetDescriptor.from_loader("zara1", Zara1Loader, Zara1Loader),
    "zara2": DatasetDescriptor.from_loader("zara2", Zara2Loader, Zara2Loader),
}

__all__ = [
    "DESCRIPTORS",
    "EthLoader",
    "HotelLoader",
    "UnivLoader",
    "Zara1Loader",
    "Zara2Loader",
]
