from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)
from dronalize.datasets.registry import DatasetDescriptor

DESCRIPTORS = {
    "eth": DatasetDescriptor.from_loader("eth", EthLoader, infer_capabilities=True),
    "hotel": DatasetDescriptor.from_loader("hotel", HotelLoader, infer_capabilities=True),
    "univ": DatasetDescriptor.from_loader("univ", UnivLoader, infer_capabilities=True),
    "zara1": DatasetDescriptor.from_loader("zara1", Zara1Loader, infer_capabilities=True),
    "zara2": DatasetDescriptor.from_loader("zara2", Zara2Loader, infer_capabilities=True),
}

__all__ = ["DESCRIPTORS", "EthLoader", "HotelLoader", "UnivLoader", "Zara1Loader", "Zara2Loader"]
