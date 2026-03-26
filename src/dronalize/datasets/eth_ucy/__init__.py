from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)

DESCRIPTORS = {
    "eth": DatasetDescriptor.from_loader("eth", EthLoader),
    "hotel": DatasetDescriptor.from_loader("hotel", HotelLoader),
    "univ": DatasetDescriptor.from_loader("univ", UnivLoader),
    "zara1": DatasetDescriptor.from_loader("zara1", Zara1Loader),
    "zara2": DatasetDescriptor.from_loader("zara2", Zara2Loader),
}

__all__ = [
    "DESCRIPTORS",
    "EthLoader",
    "HotelLoader",
    "UnivLoader",
    "Zara1Loader",
    "Zara2Loader",
]
