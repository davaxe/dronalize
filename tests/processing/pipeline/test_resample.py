# pyright: standard
from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl
import pytest
from typing_extensions import TypedDict, Unpack

from dronalize.core.functional import ResampleMethod, ResampleSpec
from dronalize.processing.pipeline.transforms import resample

if TYPE_CHECKING:
    from tests.support import DataFramePresets


class _ResampleArgs(TypedDict, total=False):
    up: int
    down: int
    coordinates: tuple[str, ...]
    max_gap: int
    sample_time: float
    emit_velocity: bool
    emit_acceleration: bool


def _all_methods_spec(**kwargs: Unpack[_ResampleArgs]) -> list[ResampleSpec]:
    add_derivatives = kwargs.get("emit_velocity", False) or kwargs.get("emit_acceleration", False)
    return [
        ResampleSpec(method=ResampleMethod.PCHIP, **kwargs),
        ResampleSpec(method=ResampleMethod.CUBIC, **kwargs),
        *([ResampleSpec(method=ResampleMethod.LINEAR, **kwargs)] if not add_derivatives else []),
    ]


def _straight_track(frames: list[int]) -> pl.DataFrame:
    return pl.DataFrame({
        "frame": frames,
        "id": [1] * len(frames),
        "x": [float(frame) for frame in frames],
        "y": [0.0] * len(frames),
        "agent_category": [1] * len(frames),
    })


@pytest.mark.parametrize("spec", _all_methods_spec(up=2, down=1, sample_time=1.0))
def test_resample_straight_track_matches_samples(spec: ResampleSpec) -> None:
    df = _straight_track([0, 1, 2])
    df_resampled = (
        resample(spec)(df.lazy())
        .collect()
        .sort("frame")
        .select("frame", "id", "x", "y", "agent_category")
    )

    assert df_resampled["frame"].to_list() == [0, 1, 2, 3, 4]
    assert df_resampled["id"].to_list() == [1, 1, 1, 1, 1]
    assert df_resampled["agent_category"].to_list() == [1, 1, 1, 1, 1]
    assert df_resampled["x"].to_list() == pytest.approx([0.0, 0.5, 1.0, 1.5, 2.0])
    assert df_resampled["y"].to_list() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0])


@pytest.mark.parametrize("spec", _all_methods_spec(up=2, down=1, max_gap=1, sample_time=1.0))
def test_resample_max_gap_splits_gaps(spec: ResampleSpec) -> None:
    df = _straight_track([0, 1, 4])
    df_resampled = resample(spec)(df.lazy()).collect().sort("frame")

    assert df_resampled["frame"].to_list() == [0, 1, 2, 8]
    assert df_resampled["x"].to_list() == pytest.approx([0.0, 0.5, 1.0, 4.0])


@pytest.mark.parametrize("spec", _all_methods_spec(up=2, down=1, max_gap=3, sample_time=1.0))
def test_resample_max_gap_keeps_allowed_gaps(spec: ResampleSpec) -> None:
    df = _straight_track([0, 1, 4])
    df_resampled = resample(spec)(df.lazy()).collect().sort("frame")

    assert df_resampled["frame"].to_list() == list(range(9))
    assert df_resampled["x"].to_list() == pytest.approx([
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
    ])


@pytest.mark.parametrize("spec", _all_methods_spec(up=2, down=1, sample_time=1.0))
def test_resample_simple(spec: ResampleSpec, scene_df_presets: DataFramePresets) -> None:
    # preset has 3 samples with no gaps
    df = scene_df_presets["single_agent"]()
    resample_transform = resample(spec)
    df_resampled = resample_transform(df.lazy()).collect()

    assert df.height == 3
    assert df_resampled.height == 5  # (3 - 1) * 2 + 1
    assert df.null_count().sum_horizontal().item() == 0


@pytest.mark.parametrize(
    "spec", _all_methods_spec(up=2, down=1, emit_velocity=True, emit_acceleration=True)
)
def test_resample_adds_derivatives(
    spec: ResampleSpec, scene_df_presets: DataFramePresets
) -> None:
    # preset has 3 samples with no gaps
    df = scene_df_presets["single_agent"]()
    resample_transform = resample(spec)
    df_resampled = resample_transform(df.lazy()).collect()

    assert df.height == 3
    assert df_resampled.height == 5  # (3 - 1) * 2 + 1
    assert df.null_count().sum_horizontal().item() == 0
    assert "vx" in df_resampled.columns
    assert "vy" in df_resampled.columns
    assert "ax" in df_resampled.columns
    assert "ay" in df_resampled.columns
