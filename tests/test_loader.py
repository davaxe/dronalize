# pyright: standard

import polars as pl
import pytest
from typing_extensions import override

from dronalize.categories import DatasetSplit
from dronalize.config import (
    BySourceSplit,
    LoaderConfig,
    SplitRequest,
    SplitWeights,
    TimeBlockSplit,
)
from dronalize.exceptions import SplitMethodNotSupportedError, SplitNotSupportedError
from dronalize.loading import (
    BaseSceneLoader,
    BaseSceneLoaderConfig,
    IngestedData,
    Source,
)
from dronalize.pipeline import Pipeline
from dronalize.scene import CANONICAL_V1, POSITIONS_ONLY_V1, SceneSchema


def _canonical_frame(n_frames: int = 1) -> pl.LazyFrame:
    frames = list(range(n_frames))
    return pl.DataFrame({
        "frame": frames,
        "id": [1] * n_frames,
        "x": [float(frame) for frame in frames],
        "y": [0.0] * n_frames,
        "vx": [0.0] * n_frames,
        "vy": [0.0] * n_frames,
        "ax": [0.0] * n_frames,
        "ay": [0.0] * n_frames,
        "yaw": [0.0] * n_frames,
        "agent_category": [0] * n_frames,
    }).lazy()


def _positions_only_scene_frame() -> pl.LazyFrame:
    return pl.DataFrame({
        "frame": [0, 1, 2],
        "id": [1, 1, 1],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "agent_category": [0, 0, 0],
    }).lazy()


class _NativeSplitLoader(BaseSceneLoader[str]):
    @classmethod
    @override
    def predefined_splits(cls) -> tuple[DatasetSplit, ...]:
        return (DatasetSplit.TRAIN, DatasetSplit.VAL, DatasetSplit.TEST)

    @override
    def sources_for_split(self, split: DatasetSplit) -> list[Source[str]]:
        if split is DatasetSplit.TRAIN:
            return [Source(identifier="train", data="train")]
        if split is DatasetSplit.VAL:
            return [Source(identifier="val", data="val")]
        if split is DatasetSplit.TEST:
            return [Source(identifier="test", data="test")]
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def ingest(self, source: Source[str]) -> list[IngestedData]:
        _ = source
        return [IngestedData(_canonical_frame())]

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline()

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=1, output_len=1, sample_time=1.0)

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return CANONICAL_V1


class _UnsplitLoader(BaseSceneLoader[str]):
    @override
    def discover_sources(self) -> list[Source[str]]:
        return [Source(identifier="only", data="only")]

    @override
    def ingest(self, source: Source[str]) -> list[IngestedData]:
        _ = source
        return [IngestedData(_canonical_frame())]

    @override
    def pipeline(self) -> Pipeline:
        return Pipeline()

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=1, output_len=1, sample_time=1.0)

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return CANONICAL_V1


class _PositionsOnlyLoader(_UnsplitLoader):
    @override
    def ingest(self, source: Source[str]) -> list[IngestedData]:
        _ = source
        return [IngestedData(_positions_only_scene_frame())]

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1


class _DefaultPipelineBlockSplitLoader(BaseSceneLoader[str]):
    config = BaseSceneLoaderConfig(block_split_enabled=True)

    @override
    def discover_sources(self) -> list[Source[str]]:
        return [Source(identifier="only", data="only")]

    @override
    def ingest(self, source: Source[str]) -> list[IngestedData]:
        _ = source
        return [IngestedData(_canonical_frame(n_frames=6))]

    @classmethod
    @override
    def default_config(cls) -> LoaderConfig:
        return LoaderConfig(input_len=1, output_len=1, sample_time=1.0)

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return CANONICAL_V1


def test_unsplit_loader_rejects_native_split_selection() -> None:
    """Loaders without native dataset partitions should reject split filtering."""
    loader = _UnsplitLoader(splits=DatasetSplit.TRAIN)

    with pytest.raises(SplitNotSupportedError):
        _ = list(loader.sources())


def test_default_pipeline_applies_block_split_request() -> None:
    """The base loader pipeline should forward block split requests to the factory."""
    unsplit_loader = _DefaultPipelineBlockSplitLoader(
        loader_config=LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
    )
    split_loader = _DefaultPipelineBlockSplitLoader(
        loader_config=LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
        split_request=SplitRequest(
            strategy=TimeBlockSplit(gap=0),
            weights=SplitWeights(train=0.5, val=0.5, test=0.0),
        ),
    )

    unsplit_processed = list(unsplit_loader.process_next(next(iter(unsplit_loader.sources()))))
    split_processed = list(split_loader.process_next(next(iter(split_loader.sources()))))

    assert len(unsplit_processed) == 1
    assert unsplit_processed[0].frame.height == 6
    assert len(split_processed) == 2
    assert [frame.frame.height for frame in split_processed] == [3, 3]


def test_loader_rejects_direct_unsupported_split_request() -> None:
    """Direct loader construction should validate unsupported custom split methods."""
    with pytest.raises(SplitMethodNotSupportedError):
        _UnsplitLoader(
            loader_config=LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
            split_request=SplitRequest(
                strategy=BySourceSplit(),
                weights=SplitWeights(train=1.0, val=0.0, test=0.0),
            ),
        )


def test_positions_only_loader_keeps_native_schema_by_default() -> None:
    """Native schemas should be preserved when no output schema is requested."""
    loader = _PositionsOnlyLoader(
        loader_config=LoaderConfig(input_len=2, output_len=1, sample_time=1.0),
        output_schema=None,
    )

    scene = next(iter(loader.scenes()))

    assert scene.schema == POSITIONS_ONLY_V1
    assert scene.inner.columns == ["frame", "id", "x", "y", "agent_category"]
    assert scene.inner["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])


def test_loader_exposes_effective_output_schema_helpers() -> None:
    """Schema helpers should reflect the effective requested output schema."""
    loader = _PositionsOnlyLoader(
        loader_config=LoaderConfig(input_len=2, output_len=1, sample_time=1.0),
        output_schema=None,
    )

    assert loader.requested_scene_schema is None
    assert loader.output_scene_schema == POSITIONS_ONLY_V1
    assert loader.requires_scene_fields("x", "y") is True
    assert loader.requires_scene_fields("vx") is False
