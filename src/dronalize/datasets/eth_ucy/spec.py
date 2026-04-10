from dronalize.config.sections import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.eth_ucy.loader import (
    EthLoader,
    HotelLoader,
    UnivLoader,
    Zara1Loader,
    Zara2Loader,
)
from dronalize.datasets.registry import DatasetSpec
from dronalize.datasets.shared.specs import (
    minimum_samples_screening,
    resample_config,
    scenes_config,
)

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        history_frames=8,
        future_frames=12,
        sample_time=0.4,
        window_step=1,
        resample=resample_config(method="linear", up=4),
    ),
    screening=minimum_samples_screening(2),
)

DATASET_SPECS = {
    "eth": DatasetSpec(
        name="eth",
        loader_factory=EthLoader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=EthLoader.native_trajectory_schema(),
        native_splits=_NATIVE_SPLITS,
    ),
    "hotel": DatasetSpec(
        name="hotel",
        loader_factory=HotelLoader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=HotelLoader.native_trajectory_schema(),
        native_splits=_NATIVE_SPLITS,
    ),
    "univ": DatasetSpec(
        name="univ",
        loader_factory=UnivLoader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=UnivLoader.native_trajectory_schema(),
        native_splits=_NATIVE_SPLITS,
    ),
    "zara1": DatasetSpec(
        name="zara1",
        loader_factory=Zara1Loader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=Zara1Loader.native_trajectory_schema(),
        native_splits=_NATIVE_SPLITS,
    ),
    "zara2": DatasetSpec(
        name="zara2",
        loader_factory=Zara2Loader.unified_factory,
        default_config=_DEFAULT_CONFIG,
        native_schema=Zara2Loader.native_trajectory_schema(),
        native_splits=_NATIVE_SPLITS,
    ),
}
