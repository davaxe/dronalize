from __future__ import annotations

import polars as pl

from dronalize.processing.pipeline.pipeline import Pipeline


def test_then_returns_new_pipeline() -> None:
    base = Pipeline()
    updated = base.then(lambda df: df.with_columns((pl.col("x") + 1).alias("x")), name="increment")

    assert base.is_empty()
    assert len(updated) == 1


def test_compose_and_rshift_preserve_order() -> None:
    first = Pipeline().then(
        lambda df: df.with_columns((pl.col("x") + 1).alias("x")), name="plus_one"
    )
    second = Pipeline().then(
        lambda df: df.with_columns((pl.col("x") * 2).alias("x")), name="times_two"
    )

    composed = first.compose(second)
    shifted = first >> second

    data = pl.DataFrame({"x": [1]})
    composed_result = next(composed.execute(data, collect=True))
    shifted_result = next(shifted.execute(data, collect=True))
    assert composed_result["x"].to_list() == [4]
    assert shifted_result["x"].to_list() == [4]


def test_execute_collect_drops_empty_frames() -> None:
    pipeline = Pipeline().then(lambda df: df.filter(pl.col("x") > 10), name="drop_all")

    collected = list(pipeline.execute(pl.DataFrame({"x": [1, 2]}), collect=True, filter_empty=True))
    unfiltered = list(
        pipeline.execute(pl.DataFrame({"x": [1, 2]}), collect=True, filter_empty=False)
    )

    assert collected == []
    assert len(unfiltered) == 1
    assert unfiltered[0].is_empty()
