# pyright: standard

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from pydantic import ValidationError
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from dronalize.core.errors import LoaderConfigError
from dronalize.processing.pipeline.functional.resample import ResampleMethod, ResampleSpec, resample


def test_spec_simplifies_ratio() -> None:
    """Resampling ratios are normalized to lowest terms."""
    spec = ResampleSpec(up=4, down=2)
    assert (spec.up, spec.down) == (2, 1)


def test_spec_rejects_derivative_lengths() -> None:
    """Derivative column groups must align with the position dimensionality."""
    with pytest.raises(ValidationError) as exc_info:
        _ = ResampleSpec(
            method=ResampleMethod.HERMITE,
            position_columns=dict.fromkeys(("x", "y")),
            input_derivatives={1: dict.fromkeys(("vx",))},
        )
    assert "must match position_columns length" in str(exc_info.value)
    assert isinstance(exc_info.value.errors()[0]["ctx"]["error"], LoaderConfigError)


def test_linear_rejects_derivatives() -> None:
    """Linear resampling rejects derivative inputs and outputs."""
    with pytest.raises(ValidationError) as exc_info:
        _ = ResampleSpec(
            method=ResampleMethod.LINEAR, output_derivatives={1: dict.fromkeys(("vx", "vy"))}
        )
    assert "does not support derivative" in str(exc_info.value)
    error = exc_info.value.errors()[0]
    assert "ctx" in error
    ctx = error["ctx"]
    assert isinstance(ctx["error"], LoaderConfigError)


def test_hermite_requires_first_order() -> None:
    """Hermite resampling requires first-order derivative constraints."""
    with pytest.raises(ValidationError) as exc_info:
        _ = ResampleSpec(method=ResampleMethod.HERMITE)
    assert "Hermite resampling requires exactly first-order derivative inputs" in str(
        exc_info.value
    )
    error = exc_info.value.errors()[0]
    assert "ctx" in error
    ctx = error["ctx"]
    assert isinstance(ctx["error"], LoaderConfigError)


def test_cubic_rejects_derivatives() -> None:
    """Cubic spline resampling no longer switches implicitly to Hermite."""
    with pytest.raises(ValidationError) as exc_info:
        _ = ResampleSpec(
            method=ResampleMethod.CUBIC, input_derivatives={1: dict.fromkeys(("vx", "vy"))}
        )
    assert "does not accept input_derivatives" in str(exc_info.value)
    assert isinstance(exc_info.value.errors()[0]["ctx"]["error"], LoaderConfigError)


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

    expected = df
    assert_frame_equal(result, expected)


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


def test_upsample_by_two_preserves_cols() -> None:
    """Linear upsampling interpolates positions and forward-fills carried columns."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "vx": [1.0, 5.0, 9.0],
        "agent_category": [5, 5, 5],
    })

    result = resample(df, ResampleSpec(up=2, down=1))

    assert result["frame"].to_list() == [0, 1, 2, 3, 4]
    assert result["x"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result["vx"].to_list() == [1.0, 1.0, 5.0, 5.0, 9.0]
    assert result["agent_category"].to_list() == [5, 5, 5, 5, 5]


def test_resample_preserves_columns() -> None:
    """Resampling implicitly carries every non-generated column."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "yaw": [0.0, 1.0, 2.0],
        "agent_category": [5, 5, 5],
    })

    result = resample(df, ResampleSpec(up=2, down=1))

    assert result.columns == ["frame", "x", "y", "yaw", "agent_category"]
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


def test_cubic_derivative_names() -> None:
    """Spline methods emit requested derivative columns with user-provided names."""
    df = pl.DataFrame({"frame": [0, 1], "x": [0.0, 1.0]})

    result = resample(
        df,
        ResampleSpec(
            up=4,
            down=1,
            method=ResampleMethod.CUBIC,
            position_columns=dict.fromkeys(("x",)),
            output_derivatives={1: dict.fromkeys(("dx",))},
        ),
    )
    x_cubic = CubicSpline(df["frame"], df["x"], bc_type="natural")
    x_evaluated = x_cubic(np.linspace(0, 1, 5))
    dx_evaluated = x_cubic(np.linspace(0, 1, 5), 1)
    assert result["frame"].to_list() == [0, 1, 2, 3, 4]
    assert result["x"].to_list() == pytest.approx(x_evaluated.tolist())
    assert result["dx"].to_list() == pytest.approx(dx_evaluated.tolist())


def test_hermite_uses_input_derivatives() -> None:
    """Hermite can reuse input derivative columns while also emitting higher orders."""
    df = pl.DataFrame({"frame": [0, 1], "x": [0.0, 1.0], "vx": [0.0, 0.0]})

    result = resample(
        df,
        ResampleSpec(
            up=4,
            down=1,
            method=ResampleMethod.HERMITE,
            position_columns=dict.fromkeys(("x",)),
            input_derivatives={1: dict.fromkeys(("vx",))},
            output_derivatives={1: dict.fromkeys(("vx",)), 2: dict.fromkeys(("ax",))},
        ),
    )

    assert result["x"].to_list() == pytest.approx([0.0, 0.15625, 0.5, 0.84375, 1.0])
    assert result["vx"].to_list() == pytest.approx([0.0, 1.125, 1.5, 1.125, 0.0])
    assert result["ax"].to_list() == pytest.approx([6.0, 3.0, 0.0, -3.0, -6.0])


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


def test_advanced_dataframe_single() -> None:
    """Test that a more complex resampling scenario matches SciPy's direct interpolation."""
    n = 40
    t = np.linspace(0, 2 * np.pi, n, dtype=np.float64)
    x = np.cos(3 * t)
    y = np.sin(4 * t + np.pi / 3)
    dx = -3 * np.sin(3 * t)
    dy = 4 * np.cos(4 * t + np.pi / 3)
    df = pl.DataFrame({"frame": np.arange(n), "x": x, "y": y, "vx": dx, "vy": dy})
    result = resample(
        df,
        ResampleSpec(
            up=4,
            down=1,
            method=ResampleMethod.HERMITE,
            position_columns=dict.fromkeys(("x", "y")),
            input_derivatives={1: dict.fromkeys(("vx", "vy"))},
            output_derivatives={1: dict.fromkeys(("vx", "vy")), 2: dict.fromkeys(("ax", "ay"))},
        ),
    )

    t = np.arange(n)
    n_new = (40 - 1) * 4 + 1
    t_new = np.arange(n_new, dtype=np.float64) * 0.25
    x_cubic = CubicHermiteSpline(t, x, dx, axis=0)
    y_cubic = CubicHermiteSpline(t, y, dy, axis=0)
    x_evaluated = x_cubic(t_new)
    y_evaluated = y_cubic(t_new)
    dx_evaluated = x_cubic(t_new, nu=1)
    dy_evaluated = y_cubic(t_new, nu=1)
    ax_evaluated = x_cubic(t_new, nu=2)
    ay_evaluated = y_cubic(t_new, nu=2)
    assert result["frame"].to_list() == list(range(4 * n - 3))
    assert np.isclose(result["x"].to_numpy(), x_evaluated).all()
    assert np.isclose(result["y"].to_numpy(), y_evaluated).all()
    assert np.isclose(result["vx"].to_numpy(), dx_evaluated).all()
    assert np.isclose(result["vy"].to_numpy(), dy_evaluated).all()
    assert np.isclose(result["ax"].to_numpy(), ax_evaluated).all()
    assert np.isclose(result["ay"].to_numpy(), ay_evaluated).all()
