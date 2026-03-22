# pyright: standard

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from dronalize.pipeline.functional.resample import ResampleMethod, ResampleSpec, resample


def test_spec_simplifies_ratio() -> None:
    """Resampling ratios are normalized to lowest terms."""
    spec = ResampleSpec(up=4, down=2)
    assert (spec.up, spec.down) == (2, 1)


def test_spec_rejects_derivative_length_mismatch() -> None:
    """Derivative column groups must align with the position dimensionality."""
    with pytest.raises(ValueError, match="must match position_columns length"):
        _ = ResampleSpec(
            method=ResampleMethod.HERMITE,
            position_columns=dict.fromkeys(("x", "y")),
            input_derivatives={1: dict.fromkeys(("vx",))},
        )


def test_linear_rejects_derivative_config() -> None:
    """Linear resampling rejects derivative inputs and outputs."""
    with pytest.raises(ValueError, match="does not support derivative"):
        _ = ResampleSpec(
            method=ResampleMethod.LINEAR,
            output_derivatives={1: dict.fromkeys(("vx", "vy"))},
        )


def test_hermite_requires_first_order_inputs() -> None:
    """Hermite resampling requires first-order derivative constraints."""
    with pytest.raises(
        ValueError, match="Hermite resampling requires exactly first-order derivative inputs"
    ):
        _ = ResampleSpec(method=ResampleMethod.HERMITE)


def test_cubic_does_not_accept_input_derivatives() -> None:
    """Cubic spline resampling no longer switches implicitly to Hermite."""
    with pytest.raises(ValueError, match="does not accept input_derivatives"):
        _ = ResampleSpec(
            method=ResampleMethod.CUBIC,
            input_derivatives={1: dict.fromkeys(("vx", "vy"))},
        )


def test_linear_no_resampling_preserves_other_columns() -> None:
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


def test_upsample_by_two_carries_rest_columns() -> None:
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


def test_resample_preserves_other_columns_implicitly() -> None:
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


def test_linear_gap_segmentation_avoids_cross_gap_interpolation() -> None:
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


def test_cubic_outputs_named_derivatives() -> None:
    """Spline methods emit requested derivative columns with user-provided names."""
    df = pl.DataFrame({
        "frame": [0, 1],
        "x": [0.0, 1.0],
    })

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

    assert result["frame"].to_list() == [0, 1, 2, 3, 4]
    assert result["x"].to_list() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert result["dx"].to_list() == pytest.approx([1.0, 1.0, 1.0, 1.0, 1.0])


def test_hermite_uses_input_derivatives_and_emits_outputs() -> None:
    """Hermite can reuse input derivative columns while also emitting higher orders."""
    df = pl.DataFrame({
        "frame": [0, 1],
        "x": [0.0, 1.0],
        "vx": [0.0, 0.0],
    })

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
