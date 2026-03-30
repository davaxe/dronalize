from dataclasses import dataclass

import polars as pl


@dataclass(slots=True, frozen=True)
class FilterContext:
    """Shared expressions and column metadata used while evaluating filter logic."""

    agent_id: str
    frame_column: str
    category_column: str | None
    scene_window: pl.Expr | list[str]
    agent_window: list[str]
    scene_start_frame: pl.Expr

    def over_scene_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the scene window to an expression."""
        return expr.over(self.scene_window)

    def over_agent_window(self, expr: pl.Expr) -> pl.Expr:
        """Apply the agent window to an expression."""
        return expr.over(self.agent_window)

    def relative_frame(self) -> pl.Expr:
        """Return the scene-relative frame index expression."""
        return pl.col(self.frame_column) - self.scene_start_frame

    def retained_agent_count(self) -> pl.Expr:
        """Return the number of retained agents in the current scene."""
        return self.over_scene_window(pl.col(self.agent_id).n_unique())

    def category_column_or_raise(self) -> str:
        """Return the category column name or raise when category cleanup is impossible."""
        if self.category_column is None:
            msg = "Category-based filtering requires a category column."
            raise ValueError(msg)
        return self.category_column
