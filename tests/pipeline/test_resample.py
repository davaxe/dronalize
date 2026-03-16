import polars as pl
from polars.testing import assert_frame_equal

from dronalize.pipeline.functional.resample import Resampling, resample


def test_no_resampling() -> None:
    """up=1, down=1 returns data unchanged."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=1, down=1))
    assert_frame_equal(result, df)


def test_downsample_by_two() -> None:
    """Downsampling by 2 keeps every other frame and re-indexes."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 10.0, 20.0, 30.0, 40.0],
    })
    result = resample(df, Resampling(up=1, down=2))
    assert result["frame"].to_list() == [0, 1, 2]
    assert result["x"].to_list() == [0.0, 2.0, 4.0]
    assert result["y"].to_list() == [0.0, 20.0, 40.0]


def test_upsample_by_two() -> None:
    """Upsampling by 2 doubles the number of frames with interpolated values."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 10.0, 20.0],
    })
    result = resample(df, Resampling(up=2, down=1))
    # 3 original points -> (3-1)*2 + 1 = 5 points after upsampling
    assert len(result) == 5
    # Intermediate values should be linearly interpolated
    x_vals = result["x"].to_list()
    assert abs(x_vals[0] - 0.0) < 1e-5
    assert abs(x_vals[1] - 1.0) < 1e-5
    assert abs(x_vals[2] - 2.0) < 1e-5
    assert abs(x_vals[3] - 3.0) < 1e-5
    assert abs(x_vals[4] - 4.0) < 1e-5


def test_fraction_simplification() -> None:
    """up=2, down=2 simplifies to 1:1 and returns data unchanged."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=2, down=2))
    assert_frame_equal(result, df)


def test_group_by_isolation() -> None:
    """Tracks belonging to different agents are resampled independently."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
        "x": [0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=1, down=2), group_by="id")
    # Each agent: 3 frames downsampled by 2 -> 2 frames each
    agent1 = result.filter(pl.col("id") == 1).sort("frame")
    agent2 = result.filter(pl.col("id") == 2).sort("frame")
    assert len(agent1) == 2
    assert len(agent2) == 2
    assert agent1["x"].to_list() == [0.0, 2.0]
    assert agent2["x"].to_list() == [10.0, 30.0]


def test_velocity_columns_use_zero_order_hold() -> None:
    """Velocity columns are preserved piecewise-constantly during upsampling."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "vx": [1.0, 5.0, 9.0],
        "vy": [0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=2, down=1), velocity_columns=("vx", "vy"))
    assert result["x"].to_list() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert result["vx"].to_list() == [1.0, 1.0, 5.0, 5.0, 9.0]
    assert result["vy"].to_list() == [0.0, 0.0, 0.0, 0.0, 0.0]


def test_acceleration_columns_use_zero_order_hold() -> None:
    """Acceleration columns are preserved piecewise-constantly during upsampling."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "ax": [1.0, 2.0, 3.0],
        "ay": [0.0, 1.0, 2.0],
    })
    result = resample(df, Resampling(up=2, down=1), acceleration_columns=("ax", "ay"))
    assert result["ax"].to_list() == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert result["ay"].to_list() == [0.0, 0.0, 1.0, 1.0, 2.0]


def test_non_position_columns_default_to_zero_order_hold() -> None:
    """Columns outside the position set default to piecewise-constant behavior."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "yaw": [0.0, 1.0, 2.0],
        "agent_category": [5, 5, 5],
    })
    result = resample(df, Resampling(up=2, down=1))
    assert result["yaw"].to_list() == [0.0, 0.0, 1.0, 1.0, 2.0]
    assert result["agent_category"].to_list() == [5, 5, 5, 5, 5]


def test_forward_fill_column() -> None:
    """Columns listed in forward_fill are forward-filled instead of interpolated."""
    df = pl.DataFrame({
        "id": [1, 1, 1],
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "agent_category": [5, 5, 5],
    })
    result = resample(df, Resampling(up=2, down=1), group_by="id")
    # All agent_category values should remain integer 5 (forward-filled, not interpolated)
    for val in result["agent_category"].to_list():
        assert val == 5


def test_lazyframe_input() -> None:
    """Resample works on a LazyFrame and returns a LazyFrame."""
    lf = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 1.0, 2.0],
        "y": [0.0, 0.0, 0.0],
    }).lazy()
    result = resample(lf, Resampling(up=1, down=1))
    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert len(collected) == 3


def test_group_by_upsample_with_velocity_columns() -> None:
    """Velocity columns are preserved independently per group during upsampling."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
        "x": [0.0, 1.0, 2.0, 0.0, 10.0, 20.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "vx": [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
    })
    result = resample(df, Resampling(up=2, down=1), group_by="id", velocity_columns=("vx",))
    agent1 = result.filter(pl.col("id") == 1).sort("frame")
    agent2 = result.filter(pl.col("id") == 2).sort("frame")
    # Each agent with 3 original points produces 5 upsampled points
    assert len(agent1) == 5
    assert len(agent2) == 5
    assert agent1["vx"].to_list() == [1.0, 1.0, 2.0, 2.0, 3.0]
    assert agent2["vx"].to_list() == [10.0, 10.0, 20.0, 20.0, 30.0]


def test_downsample_preserves_frame_column_name() -> None:
    """After downsampling the frame column still exists with re-indexed values."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4, 5],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=1, down=3))
    assert "frame" in result.columns
    assert result["frame"].to_list() == [0, 1]
    assert result["x"].to_list() == [0.0, 3.0]
