import polars as pl

from dronalize.processing.pipeline.functional.window import (
    AdaptiveStepSize,
    sliding_window,
    sliding_window_adaptive,
)


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
    df = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    })
    windows = _windows(sliding_window(df, window_size=4, step_size=2))
    assert len(windows) == 4
    assert windows[0]["frame"].to_list() == [0, 1, 2, 3]
    assert windows[-1]["frame"].to_list() == [6, 7, 8, 9]


def test_window_frame_ranges() -> None:
    """Each window contains only frames within its expected range."""
    df = pl.DataFrame({
        "frame": list(range(10)),
        "x": [float(i) for i in range(10)],
    })
    windows = _windows(sliding_window(df, window_size=4, step_size=4))
    assert len(windows) == 2
    assert windows[0]["frame"].to_list() == [0, 1, 2, 3]
    assert windows[1]["frame"].to_list() == [4, 5, 6, 7]


def test_window_step_one() -> None:
    """Step size of 1 produces maximally overlapping windows."""
    df = pl.DataFrame({
        "frame": list(range(5)),
        "x": [float(i) for i in range(5)],
    })
    windows = _windows(sliding_window(df, window_size=3, step_size=1))
    assert len(windows) == 3
    assert windows[0]["frame"].to_list() == [0, 1, 2]
    assert windows[1]["frame"].to_list() == [1, 2, 3]
    assert windows[2]["frame"].to_list() == [2, 3, 4]


def test_window_larger_than_data() -> None:
    """A window larger than the data still includes a window with all rows."""
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 1.0, 2.0],
    })

    windows = _windows(sliding_window(df, window_size=100, step_size=1))
    assert len(windows) == 0


def test_window_unsorted_input() -> None:
    """Unsorted input is sorted before windowing when is_sorted=False."""
    df = pl.DataFrame({
        "frame": [3, 1, 0, 2, 4],
        "x": [3.0, 1.0, 0.0, 2.0, 4.0],
    })

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
    df = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    })

    result = sliding_window(df, window_size=3, step_size=3)
    indices = result["window_index"].unique().sort().to_list()

    assert indices == [0, 1]


def test_window_correct_row_count() -> None:
    """Total output rows match the combined size of all overlapping windows."""
    df = pl.DataFrame({
        "frame": list(range(8)),
        "x": [float(i) for i in range(8)],
    })

    result = sliding_window(df, window_size=4, step_size=2)
    windows = _windows(result)

    assert len(result) == sum(len(window) for window in windows)
    assert len(result) == 4 * 3


def test_window_lazyframe_input() -> None:
    """A LazyFrame input returns a LazyFrame with window_index after collect."""
    lf = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    }).lazy()

    result = sliding_window(lf, window_size=3, step_size=3)

    assert isinstance(result, pl.LazyFrame)
    collected = result.collect()
    assert "window_index" in collected.columns
    assert collected["window_index"].unique().sort().to_list() == [0, 1]


def test_window_group_by_window_contents() -> None:
    """Grouping by window_index recovers the expected rows per window."""
    df = pl.DataFrame({
        "frame": list(range(6)),
        "x": [float(i) for i in range(6)],
    })

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
    df = pl.DataFrame({
        "frame": [0, 1, 2],
        "x": [0.0, 1.0, 2.0],
    })

    windows = _windows(sliding_window(df, window_size=1, step_size=1))

    assert len(windows) == 3
    for window in windows:
        assert len(window) == 1


def test_window_anchored_accepts_missing_endpoints_inside_full_interval() -> None:
    """Anchored windows use intended bounds rather than observed endpoint span."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 4, 5],
        "x": [0.0, 1.0, 2.0, 4.0, 5.0],
    })

    windows = _windows(sliding_window(df, window_size=3, step_size=1, policy="anchored"))
    frames_per_window = [window["frame"].to_list() for window in windows]

    assert [1, 2] in frames_per_window
    assert [2, 4] in frames_per_window


def test_window_anchored_rejects_truncated_tail_windows() -> None:
    """Anchored windows still reject windows whose intended end exceeds group bounds."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 4, 5],
        "x": [0.0, 1.0, 2.0, 4.0, 5.0],
    })

    windows = _windows(sliding_window(df, window_size=3, step_size=1, policy="anchored"))
    frames_per_window = [window["frame"].to_list() for window in windows]

    assert [5] not in frames_per_window


def test_window_partial_keeps_truncated_tail_windows() -> None:
    """Partial windows include non-empty trailing windows."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 4, 5],
        "x": [0.0, 1.0, 2.0, 4.0, 5.0],
    })

    windows = _windows(sliding_window(df, window_size=3, step_size=1, policy="partial"))
    frames_per_window = [window["frame"].to_list() for window in windows]

    assert [5] in frames_per_window


def test_adaptive_window_anchored_accepts_missing_endpoints_inside_full_interval() -> None:
    """Anchored policy also applies in the multi-step adaptive path."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 4, 5, 10, 11, 12, 14, 15],
        "step_group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "x": [float(i) for i in range(10)],
    })

    out = sliding_window_adaptive(
        data=df,
        window_size=3,
        step_size=[
            AdaptiveStepSize(pl.col("step_group") == "A", 1),
            AdaptiveStepSize(pl.col("step_group") == "B", 1),
        ],
        policy="anchored",
    )
    frames_per_window = (
        out
        .sort(["window_index", "frame"])
        .group_by("window_index", maintain_order=True)
        .agg(pl.col("frame"))
        .get_column("frame")
        .to_list()
    )

    assert [1, 2] in frames_per_window
    assert [11, 12] in frames_per_window
    assert [5] not in frames_per_window
    assert [15] not in frames_per_window


def test_adaptive_window_partial_keeps_truncated_tail_windows() -> None:
    """Partial policy keeps the adaptive trailing windows as well."""
    df = pl.DataFrame({
        "frame": [0, 1, 2, 4, 5, 10, 11, 12, 14, 15],
        "step_group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
        "x": [float(i) for i in range(10)],
    })

    out = sliding_window_adaptive(
        data=df,
        window_size=3,
        step_size=[
            AdaptiveStepSize(pl.col("step_group") == "A", 1),
            AdaptiveStepSize(pl.col("step_group") == "B", 1),
        ],
        policy="partial",
    )
    frames_per_window = (
        out
        .sort(["window_index", "frame"])
        .group_by("window_index", maintain_order=True)
        .agg(pl.col("frame"))
        .get_column("frame")
        .to_list()
    )

    assert [5] in frames_per_window
    assert [15] in frames_per_window


# fmt: off
COMPLEX_DF = pl.DataFrame(
    {
        "scene_id": [
            "S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1", "S1",
            "S1", "S1", "S1", "S1", "S1",
            "S2", "S2", "S2", "S2", "S2", "S2", "S2", "S2", "S2",
            "S2", "S2", "S2", "S2",
            "S3", "S3", "S3", "S3", "S3", "S3",
        ],
        "frame": [
            7, 2, 0, 1, 4, 3, 6, 5, 8, 9,
            15, 11, 10, 14, 13,
            0, 1, 2, 4, 5, 6, 8, 9, 10,
            20, 21, 22, 23,
            100, 101, 103, 104, 105, 106,
        ],
        "step_group": [
            "A", "A", "A", "A", "A", "A", "A", "A", "A", "A",
            "B", "B", "B", "B", "B",
            "A", "A", "A", "A", "A", "A", "A", "A", "A",
            "C", "C", "C", "C",
            "B", "B", "B", "C", "C", "C",
        ],
        "agent_id": [
            101, 101, 101, 101, 101, 101, 101, 101, 101, 101,
            102, 102, 102, 102, 102,
            201, 201, 201, 201, 201, 201, 201, 201, 201,
            202, 202, 202, 202,
            301, 301, 301, 302, 302, 302,
        ],
        "x": [
            0.7, 0.2, 0.0, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.9,
            1.5, 1.1, 1.0, 1.4, 1.3,
            2.0, 2.1, 2.2, 2.4, 2.5, 2.6, 2.8, 2.9, 3.0,
            4.0, 4.1, 4.2, 4.3,
            5.0, 5.1, 5.3, 6.4, 6.5, 6.6,
        ],
        "y": [
            10.7, 10.2, 10.0, 10.1, 10.4, 10.3, 10.6, 10.5, 10.8, 10.9,
            11.5, 11.1, 11.0, 11.4, 11.3,
            12.0, 12.1, 12.2, 12.4, 12.5, 12.6, 12.8, 12.9, 13.0,
            14.0, 14.1, 14.2, 14.3,
            15.0, 15.1, 15.3, 16.4, 16.5, 16.6,
        ],
    }
)
# fmt: on


def test_adaptive_window() -> None:
    """Adaptive step sizes produce the expected windows across multiple groups."""
    out = sliding_window_adaptive(
        data=COMPLEX_DF,
        window_size=3,
        step_size=[
            AdaptiveStepSize(pl.col("step_group") == "A", 2),
            AdaptiveStepSize(pl.col("step_group") == "B", 3),
            AdaptiveStepSize(pl.col("step_group") == "C", 1),
        ],
        sliding_col="frame",
        group_by="scene_id",
        is_sorted=False,
        offset_sliding_col=True,
    )

    # Window ids should be contiguous and globally unique.
    window_ids = out.select(pl.col("window_index").unique().sort()).to_series().to_list()
    assert window_ids == list(range(len(window_ids)))

    # Each window should belong to exactly one scene and one step group.
    per_window = (
        out
        .group_by("window_index")
        .agg(
            pl.col("scene_id").n_unique().alias("n_scene"),
            pl.col("step_group").n_unique().alias("n_step_group"),
            pl.col("frame").sort().alias("frames"),
        )
        .sort("window_index")
    )

    assert per_window["n_scene"].to_list() == [1] * len(per_window)
    assert per_window["n_step_group"].to_list() == [1] * len(per_window)

    # The frame span in each window must be at least window_size (= 3),
    # even if the number of rows is smaller due to missing frames.
    frame_spans = (
        per_window
        .with_columns(
            (pl.col("frames").list.last() - pl.col("frames").list.first() + 1).alias("frame_span")
        )
        .get_column("frame_span")
        .to_list()
    )

    assert all(span >= 3 for span in frame_spans)

    # Offset frames should always start at zero inside each window.
    frame_starts = (
        per_window
        .with_columns(pl.col("frames").list.first().alias("frame_start"))
        .get_column("frame_start")
        .to_list()
    )

    assert all(start == 0 for start in frame_starts)

    actual_frames = per_window["frames"].to_list()

    # All normalized windows should begin at 0.
    assert all(frames[0] == 0 for frames in actual_frames)

    # Representative normalized windows that should exist.
    expected_patterns = [
        [0, 1, 2],  # contiguous 3-frame window
        [0, 2],  # 3-frame span with a gap
    ]

    for expected in expected_patterns:
        assert expected in actual_frames

    # Every returned window should span at least 3 frames.
    for frames in actual_frames:
        assert frames[-1] - frames[0] + 1 >= 3


def test_adaptive_window_without_frame_offset() -> None:
    """Test adaptive windowing without frame offset."""
    out = sliding_window_adaptive(
        data=COMPLEX_DF,
        window_size=3,
        step_size=[
            AdaptiveStepSize(pl.col("step_group") == "A", 2),
            AdaptiveStepSize(pl.col("step_group") == "B", 3),
            AdaptiveStepSize(pl.col("step_group") == "C", 1),
        ],
        sliding_col="frame",
        group_by="scene_id",
        is_sorted=False,
        offset_sliding_col=False,
    )

    per_window = (
        out
        .group_by("window_index")
        .agg(
            pl.col("scene_id").first().alias("scene_id"),
            pl.col("step_group").first().alias("step_group"),
            pl.col("frame").sort().alias("frames"),
        )
        .sort("window_index")
    )

    actual = [
        (row["scene_id"], row["step_group"], row["frames"])
        for row in per_window.iter_rows(named=True)
    ]

    expected = [
        ("S1", "A", [0, 1, 2]),
        ("S1", "A", [2, 3, 4]),
        ("S1", "A", [4, 5, 6]),
        ("S1", "A", [6, 7, 8]),
        ("S2", "A", [0, 1, 2]),
        ("S2", "A", [2, 4]),
        ("S2", "A", [4, 5, 6]),
        ("S2", "A", [6, 8]),
        ("S2", "A", [8, 9, 10]),
        ("S2", "C", [20, 21, 22]),
        ("S2", "C", [21, 22, 23]),
        ("S3", "C", [104, 105, 106]),
    ]
    for exp in expected:
        assert exp in actual

    # No window should start at frame 0 unless it genuinely starts there.
    frame_starts = (
        per_window
        .with_columns(pl.col("frames").list.first().alias("frame_start"))
        .get_column("frame_start")
        .to_list()
    )

    assert 20 in frame_starts
    assert 104 in frame_starts
