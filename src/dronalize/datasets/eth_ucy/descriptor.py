from dronalize.config.models import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.eth_ucy.loader import EthUcyLoader
from dronalize.datasets.registry import DatasetDescriptor, DatasetSplitSupport
from dronalize.datasets.shared.presets import (
    linear_resample,
    minimum_samples_screening,
    scenes_config,
)

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        history_frames=8,
        future_frames=12,
        sample_time=0.4,
        window_step=1,
        resample=linear_resample(up=4),
    ),
    screening=minimum_samples_screening(2),
)


def _descriptor(name: str) -> DatasetDescriptor:
    return DatasetDescriptor(
        name=name,
        loader_factory=EthUcyLoader.from_loader_request,
        default_config=_DEFAULT_CONFIG,
        native_schema=EthUcyLoader.native_trajectory_schema(),
        supported_native_splits=_NATIVE_SPLITS,
        split_support=DatasetSplitSupport(scene=True, source=True),
    )


DATASET_DESCRIPTORS = {
    name: _descriptor(name) for name in ("eth_ucy", "eth", "hotel", "univ", "zara1", "zara2")
}
