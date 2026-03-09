import polars as pl
from polars.testing import assert_frame_equal

from dronalize.pipeline.ops.resample import Resampling, ResamplingMethod, resample


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


def test_add_first_derivative() -> None:
    """add_derivative=True appends velocity columns with correct values."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=1, down=1), add_derivative=True, dt=1.0)
    assert "d1_x" in result.columns
    assert "d1_y" in result.columns
    # Linear x with dt=1 -> dx/dt should be 1.0 everywhere
    for val in result["d1_x"].to_list():
        assert abs(val - 1.0) < 1e-5
    # Constant y -> dy/dt should be 0
    for val in result["d1_y"].to_list():
        assert abs(val) < 1e-5


def test_add_second_derivative() -> None:
    """add_second_derivative=True appends acceleration columns."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=1, down=1), add_second_derivative=True, dt=1.0)
    assert "d2_x" in result.columns
    assert "d2_y" in result.columns
    # Linear signal -> second derivative should be ~0 at interior points
    values = result["d2_x"].to_list()
    assert abs(values[2]) < 1e-4


def test_derivative_rename() -> None:
    """Custom derivative column names are applied correctly."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    })
    rename = {1: ["vx", "vy"]}
    result = resample(
        df, Resampling(up=1, down=1), add_derivative=True, dt=1.0, derivative_rename=rename
    )
    assert "vx" in result.columns
    assert "vy" in result.columns
    assert "d1_x" not in result.columns


def test_forward_fill_column() -> None:
    """Columns listed in forward_fill are forward-filled instead of interpolated."""
    df = pl.DataFrame({
        "id": [1, 1, 1],
        "frame": [0, 1, 2],
        "x": [0.0, 2.0, 4.0],
        "y": [0.0, 0.0, 0.0],
        "agent_category": [5, 5, 5],
    })
    result = resample(df, Resampling(up=2, down=1), group_by="id", forward_fill=["agent_category"])
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


def test_group_by_upsample_with_derivative() -> None:
    """Upsampling with derivatives under group_by produces correct per-group results."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "frame": [0, 1, 2, 0, 1, 2],
        "x": [0.0, 1.0, 2.0, 0.0, 10.0, 20.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=2, down=1), group_by="id", add_derivative=True, dt=1.0)
    agent1 = result.filter(pl.col("id") == 1).sort("frame")
    agent2 = result.filter(pl.col("id") == 2).sort("frame")
    # Each agent with 3 original points produces 5 upsampled points
    assert len(agent1) == 5
    assert len(agent2) == 5


def test_spline_method_basic() -> None:
    """Spline method produces a resampled result with the correct number of rows."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 1, 1],
        "frame": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 4.0, 9.0, 16.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    result = resample(df, Resampling(up=2, down=1, method=ResamplingMethod.SPLINE), group_by="id")
    # 5 original -> (5-1)*2/1 + 1 = 9 new points
    assert len(result) == 9


def test_spline_method_with_derivative() -> None:
    """Spline method with add_derivative produces velocity columns."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 1, 1],
        "frame": [0, 1, 2, 3, 4],
        "x": [0.0, 1.0, 2.0, 3.0, 4.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    rename = {1: ["vx", "vy"]}
    result = resample(
        df,
        Resampling(up=2, down=1, method=ResamplingMethod.SPLINE),
        group_by="id",
        add_derivative=True,
        derivative_rename=rename,
    )
    assert "vx" in result.columns
    assert "vy" in result.columns


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
