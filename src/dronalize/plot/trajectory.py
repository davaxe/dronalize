# pyright: standard

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dronalize._internal.optional import require_optional

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path
    from types import ModuleType

    import altair as alt


def plot_trajectories(
    data: pl.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    group_by: str | None = None,
    n_groups: int | None = None,
    frame_col: str = "frame",
    *,
    x_label: str | None = None,
    y_label: str | None = None,
    save_path: Path | str | None = None,
    title: str | None = None,
    group_sample_seed: int | None = None,
    highlight_frame: int | Sequence[int] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> alt.LayerChart:
    """Plot a set of trajectories from a Polars dataframe using Altair.

    Parameters
    ----------
    data : pl.DataFrame
        Underlying dataframe containing trajectory data.
    x_col : str, optional
        Column name for x-axis values. Defaults to "x".
    y_col : str, optional
        Column name for y-axis values. Defaults to "y".
    group_by : str, optional
        Column name to group trajectories by (e.g., "track_id").
    n_groups : int, optional
        Number of unique groups to plot. If None, plots all groups.
    frame_col : str, optional
        Column name representing the frame index.
    x_label : str, optional
        Label for the x-axis. If None, uses `x_col`.
    y_label : str, optional
        Label for the y-axis. If None, uses `y_col`.
    save_path : Path or str, optional
        Path to save the plot (json, html, png, etc.).
    title : str, optional
        Title for the plot.
    group_sample_seed : int, optional
        Random seed for sampling groups.
    highlight_frame : int or Sequence[int], optional
        Frame index or indices to highlight.
    **kwargs : Any
        Additional keyword arguments passed to `chart.properties()`.

    Returns
    -------
    alt.LayerChart
        An Altair LayerChart containing the trajectories and start/end markers.

    """
    alt = require_optional("altair", extra="plot")
    alt.renderers.enable("browser")
    data = _sample_trajectory_groups(
        data, group_by=group_by, n_groups=n_groups, group_sample_seed=group_sample_seed
    )

    layers = _build_trajectory_layers(
        data,
        alt=alt,
        x_col=x_col,
        y_col=y_col,
        group_by=group_by,
        frame_col=frame_col,
        x_label=x_label,
        y_label=y_label,
        highlight_frame=highlight_frame,
    )

    chart = (
        alt
        .layer(*layers)
        .properties(title=title or "", **kwargs)
        .interactive()
        .configure_axis(grid=True, gridColor="grey", gridOpacity=0.2, gridDash=[2, 2])
    )

    if save_path:
        chart.save(save_path)

    return chart


def _sample_trajectory_groups(
    data: pl.DataFrame, *, group_by: str | None, n_groups: int | None, group_sample_seed: int | None
) -> pl.DataFrame:
    """Optionally subsample grouped trajectories before plotting."""
    if n_groups is None or group_by is None:
        return data

    unique_groups = data.select(group_by).unique()
    if unique_groups.height <= n_groups:
        return data

    selected_ids = unique_groups.sample(n_groups, seed=group_sample_seed)
    return data.join(selected_ids, on=group_by, how="semi")


def _build_trajectory_layers(
    data: pl.DataFrame,
    *,
    alt: ModuleType,
    x_col: str,
    y_col: str,
    group_by: str | None,
    frame_col: str,
    x_label: str | None,
    y_label: str | None,
    highlight_frame: int | Sequence[int] | None,
) -> list[alt.Chart]:
    """Build the layered Altair charts for trajectory plotting."""
    sort_cols = [group_by, frame_col] if group_by else [frame_col]
    data = data.sort(sort_cols)

    gap_expr = pl.col(frame_col).diff().fill_null(1) > 1
    segment_expr = gap_expr.cum_sum()
    if group_by:
        segment_expr = segment_expr.over(group_by)

    data = data.with_columns(__segment=segment_expr)
    if group_by:
        start_df = data.group_by(group_by, maintain_order=True).head(1)
        end_df = data.group_by(group_by, maintain_order=True).tail(1)
    else:
        start_df = data.head(1)
        end_df = data.tail(1)

    tooltips: list[alt.Tooltip] = [
        alt.Tooltip(x_col, format=".2f"),
        alt.Tooltip(y_col, format=".2f"),
        alt.Tooltip(frame_col),
    ]
    if group_by:
        tooltips.append(alt.Tooltip(group_by))

    base = alt.Chart(data).encode(
        x=alt.X(x_col, title=x_label or x_col, scale=alt.Scale(zero=False)),
        y=alt.Y(y_col, title=y_label or y_col, scale=alt.Scale(zero=False)),
        color=alt.Color(
            group_by or alt.Undefined, scale=alt.Scale(scheme="category20"), legend=None
        ),
        order=alt.Order(field=frame_col),
        tooltip=tooltips,
    )

    gap_lines = base.mark_line(strokeDash=[4, 4], strokeWidth=1.5, opacity=0.4)

    solid_lines = base.encode(detail="__segment").mark_line(
        point=alt.OverlayMarkDef(filled=True, size=15), strokeWidth=2, opacity=0.8
    )
    start_markers = (
        alt
        .Chart(start_df)
        .mark_point(shape="circle", size=75, filled=True, fill="#2ecc71", stroke="black")
        .encode(x=x_col, y=y_col, tooltip=tooltips)
    )
    end_markers = (
        alt
        .Chart(end_df)
        .mark_point(shape="cross", size=75, filled=True, color="#e74c3c", stroke="black")
        .encode(x=x_col, y=y_col, tooltip=tooltips)
    )
    layers = [gap_lines, solid_lines, start_markers, end_markers]

    if highlight_frame is not None:
        frames_to_highlight = (
            [highlight_frame] if isinstance(highlight_frame, int) else list(highlight_frame)
        )

        highlight_df = data.filter(pl.col(frame_col).is_in(frames_to_highlight))
        if not highlight_df.is_empty():
            highlight_layer = (
                alt
                .Chart(highlight_df)
                .mark_point(
                    shape="diamond",
                    size=175,
                    filled=True,
                    fill="#FFD700",
                    stroke="black",
                    strokeWidth=1,
                )
                .encode(x=x_col, y=y_col, tooltip=tooltips)
            )
            layers.append(highlight_layer)

    return layers
