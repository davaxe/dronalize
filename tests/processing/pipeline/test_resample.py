from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from typing_extensions import TypedDict, Unpack

from dronalize.processing.pipeline.functional.resample import ResampleMethod, ResampleSpec
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


@pytest.mark.parametrize(
    "spec",
    _all_methods_spec(up=2, down=1, sample_time=1.0),
)
def test_resample_simple(spec: ResampleSpec, scene_df_presets: DataFramePresets) -> None:
    # preset has 3 samples with no gaps
    df = scene_df_presets["single_agent"]()
    resample_transform = resample(spec)
    df_resampled = resample_transform(df.lazy()).collect()

    assert df.height == 3
    assert df_resampled.height == 5  # (3 - 1) * 2 + 1
    assert df.null_count().sum_horizontal().item() == 0


@pytest.mark.parametrize(
    "spec",
    _all_methods_spec(
        up=2,
        down=1,
        emit_velocity=True,
        emit_acceleration=True,
    ),
)
def test_resample_simple_add_derivatives(
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
