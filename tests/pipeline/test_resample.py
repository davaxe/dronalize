# pyright: standard

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.interpolate import CubicSpline, PchipInterpolator

from dronalize.core.errors import LoaderConfigError
from dronalize.processing.pipeline.functional.resample import ResampleMethod, ResampleSpec, resample


def test_spec_simplifies_ratio() -> None:
    """Resampling ratios are normalized to lowest terms."""
    spec = ResampleSpec(up=4, down=2)
    assert (spec.up, spec.down) == (2, 1)


def test_linear_rejects_emitted_derivatives() -> None:
    """Linear resampling does not expose derivative outputs."""
    with pytest.raises(LoaderConfigError, match="does not support emitting derivatives"):
        _ = ResampleSpec(method=ResampleMethod.LINEAR, emit_velocity=True)


def test_linear_no_resampling_preserves_cols() -> None:
    """A no-op resample leaves non-generated columns untouched."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
        "yaw": [0.1, 0.2, 0.3],
        "agent_category": [5, 5, 5],
    })

    result = resample(df, ResampleSpec())

    assert_frame_equal(result, df)


def test_downsample_by_two() -> None:
    """Downsampling keeps every other sample and renumbers frames."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 10.0, 20.0, 30.0, 40.0],
    })

    result = resample(df, ResampleSpec(up=1, down=2))

    assert result["frame"].to_list() == [0, 1, 2]
    assert result["x"].to_list() == [0.0, 2.0, 4.0]
    assert result["y"].to_list() == [0.0, 20.0, 40.0]


def test_upsample_by_two_preserves_non_coordinate_columns() -> None:
    """Linear upsampling interpolates coordinates and forward-fills carried columns."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "yaw": [0.0, 1.0, 2.0],
        "agent_category": [5, 5, 5],
    })

    result = resample(df, ResampleSpec(up=2, down=1))

    assert result["frame"].to_list() == [0, 1, 2, 3, 4]
    assert result["x"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result["yaw"].to_list() == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert result["agent_category"].to_list() == [5, 5, 5, 5, 5]


def test_group_by_isolation() -> None:
    """Independent groups are resampled without leaking values across tracks."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
        "x": [0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })

    result = resample(df, ResampleSpec(up=1, down=2), group_by="id")

    agent1 = result.filter(pl.col("id") == 1).sort("frame")
    agent2 = result.filter(pl.col("id") == 2).sort("frame")
    assert agent1["x"].to_list() == [0.0, 2.0]
    assert agent2["x"].to_list() == [10.0, 30.0]


def test_gap_segmentation_avoids_crossing() -> None:
    """Interpolation does not bridge discontinuities larger than `max_gap`."""
    df = pl.DataFrame({
        "frame": [0, 1, 4, 5],
        "x": [0.0, 1.0, 10.0, 11.0],
        "y": [0.0, 0.0, 0.0, 0.0],
        "label": ["a", "a", "b", "b"],
    })

    result = resample(df, ResampleSpec(up=2, down=1, max_gap=1))

    assert result["frame"].to_list() == [0, 1, 2, 8, 9, 10]
    assert result["x"].to_list() == pytest.approx([0.0, 0.5, 1.0, 10.0, 10.5, 11.0])
    assert result["label"].to_list() == ["a", "a", "a", "b", "b", "b"]


def test_cubic_emits_velocity_and_acceleration() -> None:
    """Cubic resampling emits standard velocity and acceleration columns."""
    df = pl.DataFrame({"frame": [0, 1], "x": [0.0, 1.0]})

    result = resample(
        df,
        ResampleSpec(
            up=4,
            down=1,
            method=ResampleMethod.CUBIC,
            coordinates=("x",),
            emit_acceleration=True,
            emit_velocity=True,
        ),
    )
    interpolator = CubicSpline(df["frame"], df["x"], bc_type="natural")
    evaluated_x = interpolator(np.linspace(0, 1, 5))
    evaluated_vx = interpolator(np.linspace(0, 1, 5), 1)
    evaluated_ax = interpolator(np.linspace(0, 1, 5), 2)

    assert result.columns == ["frame", "x", "vx", "ax"]
    assert result["frame"].to_list() == [0, 1, 2, 3, 4]
    assert result["x"].to_list() == pytest.approx(evaluated_x.tolist())
    assert result["vx"].to_list() == pytest.approx(evaluated_vx.tolist())
    assert result["ax"].to_list() == pytest.approx(evaluated_ax.tolist())


def test_pchip_emits_velocity_columns() -> None:
    """PCHIP resampling can emit velocity columns from the interpolator."""
    df = pl.DataFrame({"frame": [0, 1, 2], "x": [0.0, 1.0, 1.5]})

    result = resample(
        df,
        ResampleSpec(
            up=2, down=1, method=ResampleMethod.PCHIP, coordinates=("x",), emit_velocity=True
        ),
    )
    interpolator = PchipInterpolator(df["frame"], df["x"])
    sample_points = np.arange(5, dtype=np.float64) * 0.5

    assert result.columns == ["frame", "x", "vx"]
    assert result["x"].to_list() == pytest.approx(interpolator(sample_points).tolist())
    assert result["vx"].to_list() == pytest.approx(interpolator(sample_points, nu=1).tolist())


def test_cubic_velocity_respects_sample_time() -> None:
    """Derivative outputs are scaled using the configured sample time."""
    df = pl.DataFrame({"frame": [0, 1], "x": [0.0, 1.0]})

    result = resample(
        df,
        ResampleSpec(
            up=2,
            down=1,
            method=ResampleMethod.CUBIC,
            coordinates=("x",),
            emit_velocity=True,
            sample_time=0.5,
        ),
    )
    interpolator = CubicSpline(np.array([0.0, 0.5]), np.array([0.0, 1.0]), bc_type="natural")
    sample_points = np.array([0.0, 0.25, 0.5])

    assert result["frame"].to_list() == [0, 1, 2]
    assert result["x"].to_list() == pytest.approx(interpolator(sample_points).tolist())
    assert result["vx"].to_list() == pytest.approx(interpolator(sample_points, 1).tolist())


def test_single_point_derivatives_are_nan() -> None:
    """Derivative outputs for single-point segments are undefined and stay NaN."""
    df = pl.DataFrame({"frame": [3], "x": [4.0]})

    result = resample(
        df,
        ResampleSpec(
            up=2,
            down=1,
            method=ResampleMethod.CUBIC,
            coordinates=("x",),
            emit_velocity=True,
            emit_acceleration=True,
        ),
    )

    assert result["frame"].to_list() == [6]
    assert result["x"].to_list() == [4.0]
    assert np.isnan(result["vx"].item())
    assert np.isnan(result["ax"].item())


def test_lazyframe_input_matches_eager() -> None:
    """Lazy and eager inputs produce the same resampled result."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "yaw": [0.0, 1.0, 2.0],
    })
    spec = ResampleSpec(up=2, down=1)

    eager = resample(df, spec)
    lazy = resample(df.lazy(), spec).collect()

    assert_frame_equal(eager, lazy)
