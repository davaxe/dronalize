from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from dronalize.common.trajectory.resample import resample_tracks
from dronalize.core._compat import require_optional

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

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
    if n_groups is not None and group_by is not None:
        unique_groups = data.select(group_by).unique()
        if unique_groups.height > n_groups:
            selected_ids = unique_groups.sample(n_groups, seed=group_sample_seed)
            data = data.join(selected_ids, on=group_by, how="semi")

    if group_by:
        # maintain_order=True is needed to identify the correct first/last rows per group
        start_df = data.group_by(group_by, maintain_order=True).head(1)
        end_df = data.group_by(group_by, maintain_order=True).tail(1)
    else:
        start_df = data.head(1)
        end_df = data.tail(1)

    base = alt.Chart(data).encode(
        x=alt.X(x_col, title=x_label or x_col),
        y=alt.Y(y_col, title=y_label or y_col),
        color=alt.Color(
            group_by or alt.Undefined, scale=alt.Scale(scheme="category20"), legend=None
        ),
    )

    tooltips: list[alt.Tooltip] = [
        alt.Tooltip(x_col, format=".2f"),
        alt.Tooltip(y_col, format=".2f"),
        alt.Tooltip(frame_col),
    ]
    if group_by:
        tooltips.append(alt.Tooltip(group_by))

    lines = base.mark_line(
        point=alt.OverlayMarkDef(filled=True, size=15),
        strokeWidth=2,
        opacity=0.8,
    ).encode(order=alt.Order(field=frame_col), tooltip=tooltips)

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

    layers = lines + start_markers + end_markers
    if highlight_frame is not None:
        # Normalize input to a list
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
            layers += highlight_layer

    chart = (
        layers
        .properties(title=title or "", **kwargs)
        .interactive()
        .configure_axis(grid=True, gridColor="grey", gridOpacity=0.2, gridDash=[2, 2])
    )

    if save_path:
        chart.save(save_path)

    return chart


def _generate_test() -> pl.DataFrame:
    t = np.linspace(0, 2 * np.pi, 50)
    frames = np.arange(len(t))
    # Track 1
    return pl.concat([
        pl.DataFrame({
            "frame": frames,
            "x": np.sin(2 * t),
            "y": np.cos(3 * t),
            "track_id": 1,
        }),
        pl.DataFrame({
            "frame": frames,
            "x": np.sin(3 * t),
            "y": np.cos(4 * t),
            "track_id": 2,
        }),
        pl.DataFrame({
            "frame": frames,
            "x": np.cos(1 * t),
            "y": np.sin(1 * t),
            "track_id": 3,
        }),
    ]).cast({
        "x": pl.Float64,
        "y": pl.Float64,
        "frame": pl.Int64,
        "track_id": pl.Int64,
    })


if __name__ == "__main__":
    # 1. Generate the complex data
    df_complex = _generate_test()
    print("Original DataFrame:")
    print(df_complex)

    df_resampled = resample_tracks(df_complex, up=3, down=1, group_by="track_id", method="spline")

    print(df_resampled.filter(pl.col("track_id") == 1))

    plot_trajectories(df_complex, group_by="track_id", n_groups=1)
    plot_trajectories(df_resampled, group_by="track_id", n_groups=1)
