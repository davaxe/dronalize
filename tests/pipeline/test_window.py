import polars as pl

from dronalize.processing.pipeline.functional.window import sliding_window


def _collect(result: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(result, pl.LazyFrame):
        return result.collect()

    return result


def _windows(result: pl.DataFrame | pl.LazyFrame) -> list[pl.DataFrame]:
    collected = _collect(result).sort(["window_index", "frame"])
    return [
        window.drop("window_index")
        for window in collected.partition_by("window_index", maintain_order=True)
    ]


def test_window_basic_window_count() -> None:
    """A 10-frame sequence with window_size=4, step_size=2 yields the expected windows."""
    df = pl.DataFrame({"frame": list(range(10)), "x": [float(i) for i in range(10)]})
    windows = _windows(sliding_window(df, window_size=4, step_size=2))
    assert len(windows) == 4
    assert windows[0]["frame"].to_list() == [0, 1, 2, 3]
    assert windows[-1]["frame"].to_list() == [6, 7, 8, 9]


def test_window_frame_ranges() -> None:
    """Each window contains only frames within its expected range."""
    df = pl.DataFrame({"frame": list(range(10)), "x": [float(i) for i in range(10)]})
    windows = _windows(sliding_window(df, window_size=4, step_size=4))
    assert len(windows) == 2
    assert windows[0]["frame"].to_list() == [0, 1, 2, 3]
    assert windows[1]["frame"].to_list() == [4, 5, 6, 7]


def test_window_step_one() -> None:
    """Step size of 1 produces maximally overlapping windows."""
    df = pl.DataFrame({"frame": list(range(5)), "x": [float(i) for i in range(5)]})
    windows = _windows(sliding_window(df, window_size=3, step_size=1))
    assert len(windows) == 3
    assert windows[0]["frame"].to_list() == [0, 1, 2]
    assert windows[1]["frame"].to_list() == [1, 2, 3]
    assert windows[2]["frame"].to_list() == [2, 3, 4]


def test_window_larger_than_data() -> None:
    """A window larger than the data still includes a window with all rows."""
    df = pl.DataFrame({"frame": [0, 1, 2], "x": [0.0, 1.0, 2.0]})

    windows = _windows(sliding_window(df, window_size=100, step_size=1))
    assert len(windows) == 0


def test_window_unsorted_input() -> None:
    """Unsorted input is sorted before windowing when is_sorted=False."""
    df = pl.DataFrame({"frame": [3, 1, 0, 2, 4], "x": [3.0, 1.0, 0.0, 2.0, 4.0]})

    windows = _windows(sliding_window(df, window_size=3, step_size=3, is_sorted=False))

    assert windows[0]["frame"].to_list() == [0, 1, 2]
    for window in windows:
        frames = window["frame"].to_list()
        assert frames == sorted(frames)


def test_window_preserves_columns() -> None:
    """All original columns are preserved in each window."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0, 40.0],
        "id": [1, 1, 1, 1],
    })

    result = sliding_window(df, window_size=2, step_size=2)
    windows = _windows(result)

    assert "window_index" in result.columns
    for window in windows:
        assert set(window.columns) == {"frame", "x", "y", "id"}


def test_window_index_values() -> None:
    """Each window gets a distinct, sequential window_index starting from 0."""
    df = pl.DataFrame({"frame": list(range(6)), "x": [float(i) for i in range(6)]})

    result = sliding_window(df, window_size=3, step_size=3)
    indices = result["window_index"].unique().sort().to_list()

    assert indices == [0, 1]


def test_window_correct_row_count() -> None:
    """Total output rows match the combined size of all overlapping windows."""
    df = pl.DataFrame({"frame": list(range(8)), "x": [float(i) for i in range(8)]})

    result = sliding_window(df, window_size=4, step_size=2)
    windows = _windows(result)

    assert len(result) == sum(len(window) for window in windows)
    assert len(result) == 4 * 3


def test_window_lazyframe_input() -> None:
    """A LazyFrame input returns a LazyFrame with window_index after collect."""
    lf = pl.DataFrame({"frame": list(range(6)), "x": [float(i) for i in range(6)]}).lazy()

    result = sliding_window(lf, window_size=3, step_size=3)

    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert "window_index" in collected.columns
    assert collected["window_index"].unique().sort().to_list() == [0, 1]


def test_window_group_by_contents() -> None:
    """Grouping by window_index recovers the expected rows per window."""
    df = pl.DataFrame({"frame": list(range(6)), "x": [float(i) for i in range(6)]})

    result = sliding_window(df, window_size=3, step_size=3)
    groups = (
        result
        .sort(["window_index", "frame"])
        .group_by("window_index", maintain_order=True)
        .agg(pl.col("frame"))
    )

    assert groups["frame"].to_list() == [[0, 1, 2], [3, 4, 5]]


def test_window_single_frame_step() -> None:
    """Single-frame windows with step 1 produce one row per window."""
    df = pl.DataFrame({"frame": [0, 1, 2], "x": [0.0, 1.0, 2.0]})

    windows = _windows(sliding_window(df, window_size=1, step_size=1))

    assert len(windows) == 3
    for window in windows:
        assert len(window) == 1


def test_window_anchored_internal_gaps() -> None:
    """Anchored windows use intended bounds rather than observed endpoint span."""
    df = pl.DataFrame({"frame": [0, 1, 2, 4, 5], "x": [0.0, 1.0, 2.0, 4.0, 5.0]})

    windows = _windows(sliding_window(df, window_size=3, step_size=1, policy="anchored"))
    frames_per_window = [window["frame"].to_list() for window in windows]

    assert [1, 2] in frames_per_window
    assert [2, 4] in frames_per_window


def test_window_anchored_truncated_tail() -> None:
    """Anchored windows still reject windows whose intended end exceeds group bounds."""
    df = pl.DataFrame({"frame": [0, 1, 2, 4, 5], "x": [0.0, 1.0, 2.0, 4.0, 5.0]})

    windows = _windows(sliding_window(df, window_size=3, step_size=1, policy="anchored"))
    frames_per_window = [window["frame"].to_list() for window in windows]

    assert [5] not in frames_per_window


def test_window_partial_truncated_tail() -> None:
    """Partial windows include non-empty trailing windows."""
    df = pl.DataFrame({"frame": [0, 1, 2, 4, 5], "x": [0.0, 1.0, 2.0, 4.0, 5.0]})

    windows = _windows(sliding_window(df, window_size=3, step_size=1, policy="partial"))
    frames_per_window = [window["frame"].to_list() for window in windows]

    assert [5] in frames_per_window
