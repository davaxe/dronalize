import polars as pl
from polars.testing import assert_frame_equal

from dronalize.common.trajectory.derivative import derivative


def test_first_derivative_linear() -> None:
    """First derivative of a linear signal (x = t) should be constant 1."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    result = derivative(df, "x", dt=1.0, n=1)
    expected_values = [1.0, 1.0, 1.0, 1.0, 1.0]
    for actual, expected in zip(result["d1_x"].to_list(), expected_values, strict=True):
        assert abs(actual - expected) < 1e-5


def test_first_derivative_quadratic() -> None:
    """First derivative of x = t^2 should approximate 2*t via central differences."""
    df = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0, 16.0]})
    result = derivative(df, "x", dt=1.0, n=1)
    # Central difference at interior points: (x[i+1] - x[i-1]) / 2
    # t=0: forward = 1, t=1: (4-0)/2=2, t=2: (9-1)/2=4, t=3: (16-4)/2=6, t=4: backward=7
    values = result["d1_x"].to_list()
    assert abs(values[1] - 2.0) < 1e-5
    assert abs(values[2] - 4.0) < 1e-5
    assert abs(values[3] - 6.0) < 1e-5


def test_second_derivative_quadratic() -> None:
    """Second derivative of x = t^2 should be approximately 2."""
    df = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0, 16.0]})
    result = derivative(df, "x", dt=1.0, n=2)
    # Only d2_x should be present (no intermediate d1_x)
    assert "d2_x" in result.columns
    assert "d1_x" not in result.columns
    # Interior points of the second derivative should be close to 2
    values = result["d2_x"].to_list()
    assert abs(values[2] - 2.0) < 1e-4


def test_second_derivative_with_intermediate() -> None:
    """With include_intermediate=True, both d1 and d2 columns should appear."""
    df = pl.DataFrame({"x": [0.0, 1.0, 4.0, 9.0, 16.0]})
    result = derivative(df, "x", dt=1.0, n=2, include_intermediate=True)
    assert "d1_x" in result.columns
    assert "d2_x" in result.columns


def test_custom_dt_scaling() -> None:
    """Setting dt=0.5 should scale the derivative by 1/0.5 = 2."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0, 4.0]})
    result_dt1 = derivative(df, "x", dt=1.0, n=1)
    result_dt05 = derivative(df, "x", dt=0.5, n=1)
    # With dt=1 the derivative of a linear signal is 1.0
    # With dt=0.5 the derivative should be 2.0
    for v1, v05 in zip(result_dt1["d1_x"].to_list(), result_dt05["d1_x"].to_list(), strict=True):
        assert abs(v05 - v1 * 2.0) < 1e-5


def test_multiple_columns() -> None:
    """Derivative can be computed for multiple columns at once."""
    df = pl.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 2.0, 4.0, 6.0],
    })
    result = derivative(df, "x", "y", dt=1.0, n=1)
    assert "d1_x" in result.columns
    assert "d1_y" in result.columns
    # Linear signals: dx/dt = 1, dy/dt = 2
    for val in result["d1_x"].to_list():
        assert abs(val - 1.0) < 1e-5
    for val in result["d1_y"].to_list():
        assert abs(val - 2.0) < 1e-5


def test_derivative_rename() -> None:
    """Custom column names via derivative_rename are applied correctly."""
    df = pl.DataFrame({
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [0.0, 0.0, 0.0, 0.0],
    })
    rename = {1: ["vx", "vy"]}
    result = derivative(df, "x", "y", dt=1.0, n=1, derivative_rename=rename)
    assert "vx" in result.columns
    assert "vy" in result.columns
    assert "d1_x" not in result.columns


def test_derivative_rename_two_orders() -> None:
    """Custom column names for both first and second derivatives."""
    df = pl.DataFrame({
        "x": [0.0, 1.0, 4.0, 9.0, 16.0],
        "y": [0.0, 0.0, 0.0, 0.0, 0.0],
    })
    rename = {1: ["vx", "vy"], 2: ["ax", "ay"]}
    result = derivative(
        df, "x", "y", dt=1.0, n=2, include_intermediate=True, derivative_rename=rename
    )
    assert "vx" in result.columns
    assert "vy" in result.columns
    assert "ax" in result.columns
    assert "ay" in result.columns


def test_group_by_isolation() -> None:
    """Derivatives are computed independently within each group."""
    df = pl.DataFrame({
        "id": [1, 1, 1, 2, 2, 2],
        "x": [0.0, 1.0, 2.0, 10.0, 20.0, 30.0],
    })
    result = derivative(df, "x", dt=1.0, n=1, group_by="id")
    values = result["d1_x"].to_list()
    # Agent 1: constant dx = 1
    assert abs(values[0] - 1.0) < 1e-5
    assert abs(values[1] - 1.0) < 1e-5
    assert abs(values[2] - 1.0) < 1e-5
    # Agent 2: constant dx = 10
    assert abs(values[3] - 10.0) < 1e-5
    assert abs(values[4] - 10.0) < 1e-5
    assert abs(values[5] - 10.0) < 1e-5


def test_group_by_list() -> None:
    """group_by accepts a list of column names."""
    df = pl.DataFrame({
        "scene": [1, 1, 1, 1],
        "id": [1, 1, 2, 2],
        "x": [0.0, 5.0, 0.0, 100.0],
    })
    result = derivative(df, "x", dt=1.0, n=1, group_by=["scene", "id"])
    values = result["d1_x"].to_list()
    # Agent 1: dx = 5
    assert abs(values[0] - 5.0) < 1e-5
    assert abs(values[1] - 5.0) < 1e-5
    # Agent 2: dx = 100
    assert abs(values[2] - 100.0) < 1e-5
    assert abs(values[3] - 100.0) < 1e-5


def test_boundary_forward_backward() -> None:
    """First and last points use forward and backward differences respectively."""
    df = pl.DataFrame({"x": [0.0, 1.0, 4.0]})
    result = derivative(df, "x", dt=1.0, n=1)
    values = result["d1_x"].to_list()
    # t=0: forward = (1-0)/1 = 1
    assert abs(values[0] - 1.0) < 1e-5
    # t=1: central = (4-0)/2 = 2
    assert abs(values[1] - 2.0) < 1e-5
    # t=2: backward = (4-1)/1 = 3
    assert abs(values[2] - 3.0) < 1e-5


def test_original_columns_preserved() -> None:
    """The original columns are not removed after computing derivatives."""
    df = pl.DataFrame({"x": [0.0, 1.0, 2.0], "y": [3.0, 4.0, 5.0]})
    result = derivative(df, "x", dt=1.0, n=1)
    assert "x" in result.columns
    assert "y" in result.columns
    assert_frame_equal(result.select("x", "y"), df)


def test_lazyframe_input() -> None:
    """Derivative works on a LazyFrame and returns a LazyFrame."""
    lf = pl.DataFrame({"x": [0.0, 1.0, 2.0, 3.0]}).lazy()
    result = derivative(lf, "x", dt=1.0, n=1)
    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert "d1_x" in collected.columns
    for val in collected["d1_x"].to_list():
        assert abs(val - 1.0) < 1e-5


def test_two_point_signal() -> None:
    """Derivative on a two-point signal uses forward and backward differences."""
    df = pl.DataFrame({"x": [0.0, 3.0]})
    result = derivative(df, "x", dt=1.0, n=1)
    values = result["d1_x"].to_list()
    assert abs(values[0] - 3.0) < 1e-5
    assert abs(values[1] - 3.0) < 1e-5
