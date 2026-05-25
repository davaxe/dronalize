from dronalize.config.models import DatasetConfig
from dronalize.core.categories import DatasetSplit
from dronalize.datasets.eth_ucy.loader import EthUcyLoader
from dronalize.datasets.registry import DatasetDescriptor, DatasetSplitSupport
from dronalize.datasets.shared.presets import (
    linear_resample,
    minimum_samples_screening,
    scenes_config,
    temporal_support,
)

_NATIVE_SPLITS = (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)
_DEFAULT_CONFIG = DatasetConfig(
    scenes=scenes_config(
        horizon_frames=20,
        default_observation_length=8,
        sample_time=0.4,
        window_step=1,
        resample=linear_resample(up=4),
    ),
    screening=minimum_samples_screening(2, required_frame=7),
)


def _descriptor(name: str) -> DatasetDescriptor:
    bounds = {
        "eth_ucy": (89, 1807),
        "eth": (89, 1440),
        "hotel": (89, 1807),
        "univ": (148, 1440),
        "zara1": (89, 1440),
        "zara2": (89, 1440),
    }
    min_frames, max_frames = bounds[name]
    return DatasetDescriptor(
        name=name,
        loader_factory=EthUcyLoader.from_loader_request,
        default_config=_DEFAULT_CONFIG,
        native_schema=EthUcyLoader.native_trajectory_schema(),
        supported_native_splits=_NATIVE_SPLITS,
        split_support=DatasetSplitSupport(scene=True, source=True),
        temporal_support=temporal_support(
            source_unit="recording",
            min_frames=min_frames,
            max_frames=max_frames,
            enabled_by_default=True,
        ),
    )


DATASET_DESCRIPTORS = {
    name: _descriptor(name) for name in ("eth_ucy", "eth", "hotel", "univ", "zara1", "zara2")
}
