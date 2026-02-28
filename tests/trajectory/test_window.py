import polars as pl

from preprocessing.common.trajectory.window import sliding_window


def test_iterable_basic_window_count() -> None:
    """A 10-frame sequence with window_size=4, step_size=2 yields the expected number of windows."""
    df = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    })
    windows = list(sliding_window(df, window_size=4, step_size=2, return_iterable=True))
    # Windows start at frames 0, 2, 4, 6 -> each covers 4 frames
    # Window at 6 covers [6,7,8,9] -> valid
    # Window at 8 covers [8,9] -> 2 rows, still non-empty
    assert len(windows) >= 4
    for w in windows:
        assert isinstance(w, pl.DataFrame)
        assert not w.is_empty()


def test_iterable_window_frame_ranges() -> None:
    """Each window contains only frames within its expected range."""
    df = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    })
    windows = list(sliding_window(df, window_size=4, step_size=4, return_iterable=True))
    # Non-overlapping: [0,1,2,3], [4,5,6,7], [8,9]
    assert windows[0]["frame"].to_list() == [0, 1, 2, 3]
    assert windows[1]["frame"].to_list() == [4, 5, 6, 7]


def test_iterable_step_one() -> None:
    """Step size of 1 produces maximally overlapping windows."""
    df = pl.DataFrame({
        "frame": list(range(5)),
        "x": [float(i) for i in range(5)],
    })
    windows = list(sliding_window(df, window_size=3, step_size=1, return_iterable=True))
    # Expect windows starting at 0, 1, 2, 3, 4
    assert len(windows) >= 3
    assert windows[0]["frame"].min() == 0
    assert windows[1]["frame"].min() == 1
    assert windows[2]["frame"].min() == 2


def test_iterable_window_larger_than_data() -> None:
    """A window larger than the data yields a single window with all rows."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 1.0, 2.0],
    })
    windows = list(sliding_window(df, window_size=100, step_size=1, return_iterable=True))
    # At least one window should contain all rows
    assert any(len(w) == 3 for w in windows)


def test_iterable_unsorted_input() -> None:
    """Unsorted input is sorted before windowing when is_sorted=False."""
    df = pl.DataFrame({
        "frame": [3, 1, 0, 2, 4],
        "x": [3.0, 1.0, 0.0, 2.0, 4.0],
    })
    windows = list(
        sliding_window(df, window_size=3, step_size=3, is_sorted=False, return_iterable=True)
    )
    # First window should start from frame 0
    assert windows[0]["frame"].min() == 0
    # Frames within each window should be sorted
    for w in windows:
        frames = w["frame"].to_list()
        assert frames == sorted(frames)


def test_iterable_preserves_columns() -> None:
    """All original columns are preserved in each window."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 3],
        "x": [0.0, 1.0, 2.0, 3.0],
        "y": [10.0, 20.0, 30.0, 40.0],
        "id": [1, 1, 1, 1],
    })
    windows = list(sliding_window(df, window_size=2, step_size=2, return_iterable=True))
    for w in windows:
        assert set(w.columns) == {"frame", "x", "y", "id"}


def test_non_iterable_returns_dataframe() -> None:
    """return_iterable=False returns a single DataFrame with a window_index column."""
    df = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    })
    result = sliding_window(df, window_size=3, step_size=3, return_iterable=False)
    assert isinstance(result, pl.DataFrame)
    assert "window_index" in result.columns


def test_non_iterable_window_index_values() -> None:
    """Each window gets a distinct, sequential window_index starting from 0."""
    df = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    })
    result = sliding_window(df, window_size=3, step_size=3, return_iterable=False)
    indices = result["window_index"].unique().sort().to_list()
    assert indices[0] == 0
    # Indices should be sequential
    for i, idx in enumerate(indices):
        assert idx == i


def test_non_iterable_correct_row_count() -> None:
    """Non-iterable mode produces the same total rows as the iterable windows combined."""
    df = pl.DataFrame({
        "frame": list(range(8)),
        "x": [float(i) for i in range(8)],
    })
    windows_iter = list(sliding_window(df, window_size=4, step_size=2, return_iterable=True))
    result_df = sliding_window(df, window_size=4, step_size=2, return_iterable=False)
    total_iterable_rows = sum(len(w) for w in windows_iter)
    assert len(result_df) == total_iterable_rows


def test_non_iterable_lazyframe_input() -> None:
    """Non-iterable mode works on a LazyFrame and returns a LazyFrame."""
    lf = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    }).lazy()
    result = sliding_window(lf, window_size=3, step_size=3, return_iterable=False)
    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert "window_index" in collected.columns
    assert len(collected) > 0


def test_iterable_from_lazyframe_collects() -> None:
    """Iterable mode on a LazyFrame collects and yields DataFrames."""
    lf = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    }).lazy()
    windows = list(sliding_window(lf, window_size=3, step_size=3, return_iterable=True))
    for w in windows:
        assert isinstance(w, pl.DataFrame)


def test_non_iterable_group_by_window_contents() -> None:
    """Grouping by window_index recovers the same data as the iterable windows."""
    df = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    })
    result = sliding_window(df, window_size=3, step_size=3, return_iterable=False)
    groups = result.group_by("window_index").agg(pl.col("frame"))
    # Each group should have at most window_size frames
    for frames in groups["frame"].to_list():
        assert len(frames) <= 3


def test_iterable_single_frame_step() -> None:
    """Single-frame windows with step 1 produce one row per window."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 1.0, 2.0],
    })
    windows = list(sliding_window(df, window_size=1, step_size=1, return_iterable=True))
    assert len(windows) == 3
    for w in windows:
        assert len(w) == 1
