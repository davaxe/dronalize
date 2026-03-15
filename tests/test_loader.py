import polars as pl
import pytest
from typing_extensions import override

from dronalize.categories import DatasetSplit
from dronalize.config import LoaderConfig
from dronalize.exceptions import SplitNotSupportedError
from dronalize.loading import BaseSceneLoader
from dronalize.loading.loader import IngestOutput, Source
from dronalize.pipeline.pipeline import Pipeline


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


class _SplitLoader(BaseSceneLoader[str]):
    @override
    def train_sources(self) -> list[Source[str]]:
        return [Source(identifier="train", inner="train")]

    @override
    def validate_sources(self) -> list[Source[str]]:
        return [Source(identifier="val", inner="val")]

    @override
    def test_sources(self) -> list[Source[str]]:
        return [Source(identifier="test", inner="test")]

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


class _ManualSplitLoader(_SplitLoader):
    @override
    def train_sources(self) -> list[Source[str]]:
        return [
            Source(
                identifier="train",
                inner="train",
                split_assignment=DatasetSplit.TEST,
            )
        ]


class _OverrideSplitLoader(_SplitLoader):
    @override
    def train_sources(self) -> list[Source[str]]:
        return [
            Source(identifier="train", inner="train").override_split_assignment(DatasetSplit.TEST)
        ]


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


def test_sources_leave_split_assignment_empty_when_none_requested() -> None:
    loader = _SplitLoader()

    assert [source.split_assignment for source in loader.sources()] == [None, None, None]


def test_scenes_leave_split_assignment_empty_when_none_requested() -> None:
    loader = _SplitLoader()

    assert [scene.split_assignment for scene in loader.scenes()] == [None, None, None]


def test_single_split_selection_only_yields_requested_split() -> None:
    loader = _SplitLoader(splits=DatasetSplit.VAL)

    assert [source.split_assignment for source in loader.sources()] == [DatasetSplit.VAL]
    assert [scene.split_assignment for scene in loader.scenes()] == [DatasetSplit.VAL]


def test_unsplit_loader_leaves_split_assignment_empty() -> None:
    loader = _UnsplitLoader()

    assert [source.split_assignment for source in loader.sources()] == [None]
    assert [scene.split_assignment for scene in loader.scenes()] == [None]


def test_unsplit_loader_rejects_predefined_split_selection() -> None:
    loader = _UnsplitLoader(splits=DatasetSplit.TRAIN)

    with pytest.raises(SplitNotSupportedError):
        _ = list(loader.sources())


def test_explicit_source_override_wins_over_inferred_split() -> None:
    loader = _OverrideSplitLoader(splits=DatasetSplit.TRAIN)

    assert [source.split_assignment for source in loader.sources()] == [DatasetSplit.TEST]
    assert [scene.split_assignment for scene in loader.scenes()] == [DatasetSplit.TEST]


def test_inferred_split_wins_without_explicit_override() -> None:
    loader = _ManualSplitLoader(splits=DatasetSplit.TRAIN)

    assert [source.split_assignment for source in loader.sources()] == [DatasetSplit.TRAIN]
    assert [scene.split_assignment for scene in loader.scenes()] == [DatasetSplit.TRAIN]
