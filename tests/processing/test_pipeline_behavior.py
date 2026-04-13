from __future__ import annotations

import polars as pl
import pytest

from dronalize.processing.pipeline import Pipeline


def test_pipeline_is_immutable_on_then() -> None:
    base = Pipeline()
    updated = base.then(lambda df: df.with_columns((pl.col("x") + 1).alias("x")), name="increment")

    assert base.is_empty()
    assert len(updated) == 1


def test_pipeline_compose_and_rshift_preserve_step_order() -> None:
    first = Pipeline().then(
        lambda df: df.with_columns((pl.col("x") + 1).alias("x")), name="plus_one"
    )
    second = Pipeline().then(
        lambda df: df.with_columns((pl.col("x") * 2).alias("x")), name="times_two"
    )

    composed = first.compose(second)
    shifted = first >> second

    data = pl.DataFrame({"x": [1]})
    assert composed.execute_single(data, collect=True)["x"].to_list() == [4]
    assert shifted.execute_single(data, collect=True)["x"].to_list() == [4]


def test_execute_single_rejects_flat_map_outputs() -> None:
    pipeline = Pipeline().then_flat_map(lambda df: [df, df], name="duplicate")

    with pytest.raises(ValueError, match="expected exactly 1 output"):
        _ = pipeline.execute_single(pl.DataFrame({"x": [1]}), collect=True)


def test_execute_collect_filters_empty_frames_by_default() -> None:
    pipeline = Pipeline().then(lambda df: df.filter(pl.col("x") > 10), name="drop_all")

    collected = list(pipeline.execute(pl.DataFrame({"x": [1, 2]}), collect=True, filter_empty=True))
    unfiltered = list(
        pipeline.execute(pl.DataFrame({"x": [1, 2]}), collect=True, filter_empty=False)
    )

    assert collected == []
    assert len(unfiltered) == 1
    assert unfiltered[0].is_empty()
