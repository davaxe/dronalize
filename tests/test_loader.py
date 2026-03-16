# pyright: standard

import polars as pl
import pytest
from typing_extensions import override

from dronalize.categories import DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.exceptions import SplitNotSupportedError
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.pipeline.pipeline import Pipeline
from dronalize.scene import CANONICAL_V1, POSITIONS_ONLY_V1, SceneSchema


def _scene_frame() -> pl.LazyFrame:
    return pl.DataFrame({
        "frame": [0],
        "id": [1],
        "x": [0.0],
        "y": [0.0],
        "vx": [0.0],
        "vy": [0.0],
        "ax": [0.0],
        "ay": [0.0],
        "yaw": [0.0],
        "agent_category": [0],
    }).lazy()


def _positions_only_scene_frame() -> pl.LazyFrame:
    return pl.DataFrame({
        "frame": [0, 1, 2],
        "id": [1, 1, 1],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "agent_category": [0, 0, 0],
    }).lazy()


class _SplitLoader(BaseSceneLoader[str]):
    @override
    def sources_for_split(self, split: DatasetSplit) -> list[Source[str]]:
        if split is DatasetSplit.TRAIN:
            return [Source(identifier="train", inner="train")]
        if split is DatasetSplit.VAL:
            return [Source(identifier="val", inner="val")]
        if split is DatasetSplit.TEST:
            return [Source(identifier="test", inner="test")]
        raise SplitNotSupportedError(type(self).__name__, split)

    @override
    def ingest(self, source: Source[str]) -> list[IngestOutput]:
        _ = source
        return [(_scene_frame(), None)]

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


class _ManualSplitLoader(_SplitLoader):
    @override
    def sources_for_split(self, split: DatasetSplit) -> list[Source[str]]:
        if split is DatasetSplit.TRAIN:
            return [
                Source(
                    identifier="train",
                    inner="train",
                    split_assignment=DatasetSplit.TEST,
                )
            ]
        return super().sources_for_split(split)


class _OverrideSplitLoader(_SplitLoader):
    @override
    def sources_for_split(self, split: DatasetSplit) -> list[Source[str]]:
        if split is DatasetSplit.TRAIN:
            return [
                Source(
                    identifier="train",
                    inner="train",
                ).override_split_assignment(DatasetSplit.TEST)
            ]
        return super().sources_for_split(split)


class _UnsplitLoader(BaseSceneLoader[str]):
    @override
    def discover_sources(self) -> list[Source[str]]:
        return [Source(identifier="only", inner="only")]

    @override
    def ingest(self, source: Source[str]) -> list[IngestOutput]:
        _ = source
        return [(_scene_frame(), None)]

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
    def ingest(self, source: Source[str]) -> list[IngestOutput]:
        _ = source
        return [(_positions_only_scene_frame(), None)]

    @classmethod
    @override
    def native_scene_schema(cls) -> SceneSchema:
        return POSITIONS_ONLY_V1


def test_sources_leave_split_assignment_empty_when_none_requested() -> None:
    """Test that if no splits are requested, the split assignment for sources is None."""
    loader = _SplitLoader()

    assert [source.split_assignment for source in loader.sources()] == [
        DatasetSplit.TRAIN,
        DatasetSplit.VAL,
        DatasetSplit.TEST,
    ]


def test_single_split_selection_only_yields_requested_split() -> None:
    """Test that a single split selection works."""
    loader = _SplitLoader(splits=DatasetSplit.VAL)

    assert [source.split_assignment for source in loader.sources()] == [DatasetSplit.VAL]
    assert [scene.split_assignment for scene in loader.scenes()] == [DatasetSplit.VAL]


def test_unsplit_loader_leaves_split_assignment_empty() -> None:
    """Test that split assingment is None when the loader does not support splits."""
    loader = _UnsplitLoader()

    assert [source.split_assignment for source in loader.sources()] == [None]
    assert [scene.split_assignment for scene in loader.scenes()] == [None]


def test_unsplit_loader_rejects_predefined_split_selection() -> None:
    """Test that splits that are not supported by the loader raise an exception."""
    loader = _UnsplitLoader(splits=DatasetSplit.TRAIN)

    with pytest.raises(SplitNotSupportedError):
        _ = list(loader.sources())


def test_explicit_source_override_wins_over_inferred_split() -> None:
    """Test override behavior."""
    loader = _OverrideSplitLoader(splits=DatasetSplit.TRAIN)

    assert [source.split_assignment for source in loader.sources()] == [DatasetSplit.TEST]
    assert [scene.split_assignment for scene in loader.scenes()] == [DatasetSplit.TEST]


def test_inferred_split_wins_without_explicit_override() -> None:
    """Test that if no explicit override is provided, the inferred split assignment is used."""
    loader = _ManualSplitLoader(splits=DatasetSplit.TRAIN)

    assert [source.split_assignment for source in loader.sources()] == [DatasetSplit.TEST]
    assert [scene.split_assignment for scene in loader.scenes()] == [DatasetSplit.TEST]


def test_positions_only_loader_keeps_native_schema_by_default() -> None:
    """Test that native positions-only loaders keep their native schema by default."""
    loader = _PositionsOnlyLoader(
        loader_config=LoaderConfig(input_len=2, output_len=1, sample_time=1.0)
    )

    scene = next(iter(loader.scenes()))

    assert scene.schema == POSITIONS_ONLY_V1
    assert scene.inner.columns == ["frame", "id", "x", "y", "agent_category"]
    assert scene.inner["x"].to_list() == pytest.approx([0.0, 1.0, 2.0])
