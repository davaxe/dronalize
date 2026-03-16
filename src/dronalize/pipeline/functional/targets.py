from __future__ import annotations

import polars as pl


def target_candidates(
    data: pl.DataFrame,
    input_len: int,
    *,
    min_input_frames: int = 1,
    min_output_frames: int = 1,
) -> list[int]:
    """Return candidate target agents that have sufficient valid frames.

    An agent qualifies if it has at least `min_input_frames` observations
    in the input window and at least `min_output_frames` in the output window.
    """
    start_frame = data["frame"].min()
    return (
        data
        .with_columns((pl.col("frame") - start_frame).alias("t"))
        .group_by("id")
        .agg(
            input_frames=pl.col("t").filter(pl.col("t") < input_len).n_unique(),
            output_frames=pl.col("t").filter(pl.col("t") >= input_len).n_unique(),
        )
        .filter(
            (pl.col("input_frames") >= min_input_frames)
            & (pl.col("output_frames") >= min_output_frames)
        )
        .select("id")
        .to_series()
        .to_list()
    )
