"""Lane-change-aware pipeline extensions for highway-style datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from dronalize.processing.pipeline import transforms as tr
from dronalize.processing.pipeline.contributions import PipelineStage, StageContributions
from dronalize.processing.pipeline.pipeline import Pipeline

if TYPE_CHECKING:
    from dronalize.processing.pipeline.context import BuildContext

_LANE_CHANGE_EVENT_COLUMN = "valid_lane_change"
_SCENE_LANE_CHANGE_COUNT_COLUMN = "_scene_lane_change_count"


@dataclass(frozen=True, slots=True)
class LaneChangeSamplingExtension:
    """Keep lane-change scenes and deterministically thin no-change scenes."""

    lane_id_column: str = "lane_id"
    negative_keep_every: int = 3
    min_lane_change_events: int = 1
    persist: int = 1
    margin_before: int = 0
    margin_after: int = 0

    def __post_init__(self) -> None:
        """Validate parameters."""
        if self.negative_keep_every < 1:
            msg = "negative_keep_every must be at least 1."
            raise ValueError(msg)
        if self.min_lane_change_events < 1:
            msg = "min_lane_change_events must be at least 1."
            raise ValueError(msg)

    def compile(self, ctx: BuildContext) -> StageContributions:
        """Compile lane-change transforms against an immutable build context."""
        if not ctx.has_window:
            msg = "LaneChangeSamplingExtension requires window sampling to be enabled."
            raise ValueError(msg)

        pre = Pipeline().then(
            tr.valid_lane_change(
                persist=self.persist,
                margin_before=self.margin_before,
                margin_after=self.margin_after,
                frame_column=ctx.frame_column,
                agent_id_column=ctx.agent_id_column,
                lane_id_column=self.lane_id_column,
                group_by=list(ctx.window_group_columns) or None,
                valid_column=_LANE_CHANGE_EVENT_COLUMN,
            )
        )

        scene_id_column = ctx.scene_id_column
        if scene_id_column is None:
            msg = "LaneChangeSamplingExtension requires a scene ID column in the build context."
            raise ValueError(msg)

        def _label_scenes(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(
                pl
                .col(_LANE_CHANGE_EVENT_COLUMN)
                .sum()
                .over(scene_id_column)
                .alias(_SCENE_LANE_CHANGE_COUNT_COLUMN)
            )

        def _sample_scenes(df: pl.LazyFrame) -> pl.LazyFrame:
            is_positive = (
                pl.col(_SCENE_LANE_CHANGE_COUNT_COLUMN).fill_null(value=0)
                >= self.min_lane_change_events
            )
            keep_negative = (
                pl.lit(value=True)
                if self.negative_keep_every == 1
                else (pl.col(scene_id_column) % self.negative_keep_every) == 0
            )
            return df.filter(is_positive | keep_negative).select(
                pl.all().exclude(_LANE_CHANGE_EVENT_COLUMN, _SCENE_LANE_CHANGE_COUNT_COLUMN)
            )

        post = (
            Pipeline()
            .then(_label_scenes, name="label_lane_change_scenes")
            .then(_sample_scenes, name="sample_lane_change_scenes")
        )

        return StageContributions(
            transforms={PipelineStage.PRE_WINDOW: pre, PipelineStage.POST_SCREENING: post},
            require_scene_id=True,
        )
