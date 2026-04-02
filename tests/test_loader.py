# pyright: standard

import polars as pl
import pytest
from typing_extensions import override

from dronalize.core.categories import DatasetSplit
from dronalize.core.errors import SplitNotSupportedError, SplitStrategyNotSupportedError
from dronalize.core.scene import CANONICAL, POSITIONS_ONLY, SceneSchema
from dronalize.processing.ingest import (
    BaseSceneLoader,
    BySourceSplit,
    IngestedData,
    LoaderConfig,
    LoaderSplitCapabilities,
    Source,
    SplitConfig,
    SplitWeights,
    TimeBlockSplit,
)
from dronalize.processing.pipeline import Pipeline


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
        return CANONICAL


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
        return CANONICAL


class _PositionsOnlyLoader(_UnsplitLoader):
    @override
    def ingest(self, source: Source[str]) -> list[IngestedData]:
        _ = source
        return [IngestedData(_positions_only_scene_frame())]

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY


class _DefaultPipelineBlockSplitLoader(BaseSceneLoader[str]):
    split_capabilities = LoaderSplitCapabilities(supports_block_split=True)

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
        return CANONICAL


def test_unsplit_loader_rejects_splits() -> None:
    """Loaders without native dataset partitions should reject split filtering."""
    loader = _UnsplitLoader(splits=DatasetSplit.TRAIN)

    with pytest.raises(SplitNotSupportedError):
        _ = list(loader.sources())


def test_default_pipeline_block_split() -> None:
    """The base loader pipeline should forward block split requests to the factory."""
    unsplit_loader = _DefaultPipelineBlockSplitLoader(
        loader_config=LoaderConfig(input_len=1, output_len=1, sample_time=1.0)
    )
    split_loader = _DefaultPipelineBlockSplitLoader(
        loader_config=LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
        split_request=SplitConfig(
            mode=TimeBlockSplit(gap=0), ratio=SplitWeights(train=0.5, val=0.5, test=0.0)
        ),
    )

    unsplit_processed = list(unsplit_loader.process_next(next(iter(unsplit_loader.sources()))))
    split_processed = list(split_loader.process_next(next(iter(split_loader.sources()))))

    assert len(unsplit_processed) == 1
    assert unsplit_processed[0].frame.height == 6
    assert len(split_processed) == 2
    assert [frame.frame.height for frame in split_processed] == [3, 3]


def test_loader_rejects_bad_split_request() -> None:
    """Direct loader construction should validate unsupported custom split modes."""
    with pytest.raises(SplitStrategyNotSupportedError):
        _UnsplitLoader(
            loader_config=LoaderConfig(input_len=1, output_len=1, sample_time=1.0),
            split_request=SplitConfig(
                mode=BySourceSplit(), ratio=SplitWeights(train=1.0, val=0.0, test=0.0)
            ),
        )


def test_positions_only_loader_native_schema() -> None:
    """Native schemas should be preserved when no output schema is requested."""
    loader = _PositionsOnlyLoader(
        loader_config=LoaderConfig(input_len=2, output_len=1, sample_time=1.0), output_schema=None
    )

    scene = next(iter(loader.scenes()))

    assert scene.schema == POSITIONS_ONLY
    assert scene.frame.columns == ["frame", "id", "x", "y", "agent_category"]
    assert scene.frame["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])


def test_loader_output_schema_helpers() -> None:
    """Schema helpers should reflect the effective requested output schema."""
    loader = _PositionsOnlyLoader(
        loader_config=LoaderConfig(input_len=2, output_len=1, sample_time=1.0), output_schema=None
    )

    assert loader.requested_scene_schema is None
    assert loader.output_scene_schema == POSITIONS_ONLY
    assert loader.requires_scene_fields("x", "y") is True
    assert loader.requires_scene_fields("vx") is False
