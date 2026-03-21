from dronalize.datasets._registry import DatasetDescriptor
from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)

DESCRIPTORS = {
    "eth": DatasetDescriptor(
        name="eth",
        loader_factory=EthLoader,
        default_config=EthLoader.default_config(),
        default_map_config=EthLoader.default_map_config(),
        has_map=False,
        predefined_splits=list(EthLoader.predefined_splits()),
    ),
    "hotel": DatasetDescriptor(
        name="hotel",
        loader_factory=HotelLoader,
        default_config=HotelLoader.default_config(),
        default_map_config=HotelLoader.default_map_config(),
        has_map=False,
        predefined_splits=list(HotelLoader.predefined_splits()),
    ),
    "univ": DatasetDescriptor(
        name="univ",
        loader_factory=UnivLoader,
        default_config=UnivLoader.default_config(),
        default_map_config=UnivLoader.default_map_config(),
        has_map=False,
        predefined_splits=list(UnivLoader.predefined_splits()),
    ),
    "zara1": DatasetDescriptor(
        name="zara1",
        loader_factory=Zara1Loader,
        default_config=Zara1Loader.default_config(),
        default_map_config=Zara1Loader.default_map_config(),
        has_map=False,
        predefined_splits=list(Zara1Loader.predefined_splits()),
    ),
    "zara2": DatasetDescriptor(
        name="zara2",
        loader_factory=Zara2Loader,
        default_config=Zara2Loader.default_config(),
        default_map_config=Zara2Loader.default_map_config(),
        has_map=False,
        predefined_splits=list(Zara2Loader.predefined_splits()),
    ),
}

__all__ = [
    "DESCRIPTORS",
    "EthLoader",
    "HotelLoader",
    "UnivLoader",
    "Zara1Loader",
    "Zara2Loader",
]
