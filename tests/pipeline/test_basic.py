import math

import polars as pl

from dronalize.pipeline.functional.basic import collect, lazy, yaw_from_pos, yaw_from_vel


def test_lazy_from_dataframe() -> None:
    """Converting a DataFrame to lazy returns a LazyFrame."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = lazy(df)
    assert isinstance(result, pl.LazyFrame)


def test_lazy_from_lazyframe_is_noop() -> None:
    """Calling lazy on a LazyFrame returns the same object."""
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    result = lazy(lf)
    assert result is lf


def test_collect_from_lazyframe() -> None:
    """Collecting a LazyFrame returns a DataFrame."""
    lf = pl.DataFrame({"a": [1, 2, 3]}).lazy()
    result = collect(lf)
    assert isinstance(result, pl.DataFrame)


def test_collect_from_dataframe_is_noop() -> None:
    """Calling collect on a DataFrame returns the same object."""
    df = pl.DataFrame({"a": [1, 2, 3]})
    result = collect(df)
    assert result is df


def test_yaw_from_vel_east() -> None:
    """Velocity pointing east (positive x) should give yaw ≈ 0."""
    df = pl.DataFrame({"vx": [1.0], "vy": [0.0]})
    result = yaw_from_vel(df)
    assert abs(result["yaw"][0]) < 1e-6


def test_yaw_from_vel_north() -> None:
    """Velocity pointing north (positive y) should give yaw ≈ π/2."""
    df = pl.DataFrame({"vx": [0.0], "vy": [1.0]})
    result = yaw_from_vel(df)
    assert abs(result["yaw"][0] - math.pi / 2) < 1e-6


def test_yaw_from_vel_west() -> None:
    """Velocity pointing west (negative x) should give yaw ≈ π."""
    df = pl.DataFrame({"vx": [-1.0], "vy": [0.0]})
    result = yaw_from_vel(df)
    assert abs(abs(result["yaw"][0]) - math.pi) < 1e-6


def test_yaw_from_vel_custom_columns() -> None:
    """Custom column names are respected."""
    df = pl.DataFrame({"u": [0.0], "v": [1.0]})
    result = yaw_from_vel(df, vx_col="u", vy_col="v", yaw_col="heading")
    assert "heading" in result.columns
    assert abs(result["heading"][0] - math.pi / 2) < 1e-6


def test_yaw_from_vel_multiple_rows() -> None:
    """Yaw is computed independently for each row."""
    df = pl.DataFrame({
        "vx": [1.0, 0.0, -1.0],
        "vy": [0.0, 1.0, 0.0],
    })
    result = yaw_from_vel(df)
    expected_yaw = [0.0, math.pi / 2, math.pi]
    for actual, expected in zip(result["yaw"].to_list(), expected_yaw, strict=True):
        assert abs(actual - expected) < 1e-6


def test_yaw_from_vel_on_lazyframe() -> None:
    """yaw_from_vel works on a LazyFrame and returns a LazyFrame."""
    lf = pl.DataFrame({"vx": [1.0], "vy": [0.0]}).lazy()
    result = yaw_from_vel(lf)
    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert abs(collected["yaw"][0]) < 1e-6


def test_yaw_from_pos_known_direction() -> None:
    """Position moving east gives yaw ≈ 0 (after the first null from diff)."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0]})
    result = yaw_from_pos(df)
    assert abs(result["yaw"][0]) < 1e-6
    assert abs(result["yaw"][1]) < 1e-6
    assert abs(result["yaw"][2]) < 1e-6


def test_yaw_from_pos_diagonal() -> None:
    """Moving diagonally northeast gives yaw ≈ π/4."""
    df = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 1.0]})
    result = yaw_from_pos(df)
    assert abs(result["yaw"][0] - math.pi / 4) < 1e-6
    assert abs(result["yaw"][1] - math.pi / 4) < 1e-6


def test_yaw_from_pos_custom_columns() -> None:
    """Custom column names are respected."""
    df = pl.DataFrame({"px": [0.0, 0.0], "py": [0.0, 1.0]})
    result = yaw_from_pos(df, x_col="px", y_col="py", yaw_col="heading")
    assert "heading" in result.columns
    assert abs(result["heading"][0] - math.pi / 2) < 1e-6
    assert abs(result["heading"][1] - math.pi / 2) < 1e-6


def test_yaw_from_pos_on_lazyframe() -> None:
    """yaw_from_pos works on a LazyFrame and returns a LazyFrame."""
    lf = pl.DataFrame({"x": [0.0, 1.0], "y": [0.0, 0.0]}).lazy()
    result = yaw_from_pos(lf)
    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert abs(collected["yaw"][0]) < 1e-6
    assert abs(collected["yaw"][1]) < 1e-6
