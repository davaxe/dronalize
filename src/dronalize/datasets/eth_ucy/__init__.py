from dronalize.datasets.eth_ucy.loader import (
    EthLoader as _EthLoader,
)
from dronalize.datasets.eth_ucy.loader import (
    HotelLoader as _HotelLoader,
)
from dronalize.datasets.eth_ucy.loader import (
    UnivLoader as _UnivLoader,
)
from dronalize.datasets.eth_ucy.loader import (
    Zara1Loader as _Zara1Loader,
)
from dronalize.datasets.eth_ucy.loader import (
    Zara2Loader as _Zara2Loader,
)
from dronalize.datasets.registry import DatasetSpec

DATASET_SPECS = {
    "eth": DatasetSpec.from_loader("eth", _EthLoader, infer_capabilities=True),
    "hotel": DatasetSpec.from_loader("hotel", _HotelLoader, infer_capabilities=True),
    "univ": DatasetSpec.from_loader("univ", _UnivLoader, infer_capabilities=True),
    "zara1": DatasetSpec.from_loader("zara1", _Zara1Loader, infer_capabilities=True),
    "zara2": DatasetSpec.from_loader("zara2", _Zara2Loader, infer_capabilities=True),
}

__all__ = ["DATASET_SPECS"]
