import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import seaborn as sns

from preprocessing.trajectory.resample import resample_tracks


def plot_comparison(
    original_df: pl.DataFrame,
    resampled_df: pl.DataFrame,
    group_by: str = "track_id",
    n_groups: int | None = None,
):
    # 1. Filter logic
    if n_groups is not None:
        selected_ids = original_df.select(group_by).unique().head(n_groups)
        original_df = original_df.join(selected_ids, on=group_by, how="semi")
        resampled_df = resampled_df.join(selected_ids, on=group_by, how="semi")

    # 2. Create a consistent color mapping
    # This ensures Track 5 is the same color in both plots
    unique_ids = original_df.select(group_by).unique().to_series().to_list()

    # "tab10" is high contrast for few items; "husl" is distinct for many items
    palette_name = "tab10" if len(unique_ids) <= 10 else "husl"
    colors = sns.color_palette(palette_name, n_colors=len(unique_ids))
    color_map = dict(zip(unique_ids, colors, strict=True))

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=True, sharey=True)
    dataframes = [original_df.to_pandas(), resampled_df.to_pandas()]
    titles = ["Original (Low Frequency)", "Resampled (Spline Interpolation)"]

    for i, (df, ax) in enumerate(zip(dataframes, axes, strict=True)):
        sns.lineplot(
            data=df,
            x="x",
            y="y",
            hue=group_by,
            ax=ax,
            palette=color_map,  # Use the fixed dictionary here
            alpha=0.8,  # Increased opacity for better visibility
            linewidth=2,  # Slightly thicker lines
            marker="o",
            markersize=4,
            markeredgecolor="white",  # Adds a tiny border to points to pop against background
            markeredgewidth=0.5,
            sort=False,
            legend=False,
        )

        # Plot Start/End markers
        unique_groups = df[group_by].unique()
        for tid in unique_groups:
            subset = df[df[group_by] == tid]
            if len(subset) == 0:
                continue

            # Start: Green Circle with black edge for contrast
            ax.plot(
                subset["x"].iloc[0],
                subset["y"].iloc[0],
                marker="o",
                color="#2ecc71",
                markersize=8,
                markeredgecolor="black",
                label="Start" if tid == unique_groups[0] and i == 0 else "",
                zorder=10,
            )
            # End: Red X with black edge
            ax.plot(
                subset["x"].iloc[-1],
                subset["y"].iloc[-1],
                marker="X",
                color="#e74c3c",
                markersize=8,
                markeredgecolor="black",
                label="End" if tid == unique_groups[0] and i == 0 else "",
                zorder=10,
            )

        ax.set_title(titles[i], fontsize=12, pad=15, fontweight="bold")
        ax.grid(visible=True, linestyle="--", alpha=0.3)
        ax.set_xlabel("X Position")

    axes[0].set_ylabel("Y Position")

    # Add single legend
    if n_groups is not None and n_groups > 0:
        fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=True)

    plt.tight_layout()
    plt.show()


def plot_overlaid_comparison(
    *dfs: pl.DataFrame,
    labels: list[str] | None = None,
    group_by: str = "track_id",
    n_groups: int | None = None,
):
    if labels is not None and len(dfs) != len(labels):
        msg = f"Length of labels ({len(labels)}) must match number of DataFrames ({len(dfs)})."
        raise ValueError(msg)

    plt.figure(figsize=(12, 10))
    ax = plt.gca()

    # 1. Define distinct line styles for each dataframe type
    # This helps distinguish 'Original' from 'Resampled' even if they are the same color
    line_styles = ["-", "--", ":", "-."]

    # 2. Get unique track IDs for consistent coloring
    all_ids = []
    for df in dfs:
        all_ids.extend(df.select(group_by).unique().to_series().to_list())
    unique_ids = sorted(set(all_ids))

    if n_groups is not None:
        unique_ids = unique_ids[:n_groups]

    # Map colors to Track IDs
    palette = sns.color_palette("husl", n_colors=len(unique_ids))
    color_map = dict(zip(unique_ids, palette, strict=True))

    # 3. Plotting Loop
    for df_idx, (df_pl, label) in enumerate(
        zip(dfs, labels or [None] * len(dfs), strict=True)
    ):
        df = df_pl.filter(pl.col(group_by).is_in(unique_ids)).to_pandas()
        style = line_styles[df_idx % len(line_styles)]

        for tid in unique_ids:
            subset = df[df[group_by] == tid]
            if subset.empty:
                continue

            # Plot the line
            # We only add the label for the first track of each DF to avoid legend bloat
            plot_label = label if tid == unique_ids[0] else ""

            ax.plot(
                subset["x"],
                subset["y"],
                linestyle=style,
                color=color_map[tid],
                label=plot_label,
                linewidth=2 if df_idx == 0 else 1.5,
                alpha=0.7,
                marker="o" if df_idx == 0 else None,  # Markers only for the first DF
                markersize=4,
            )

            # Special markers for the 'Master' or first dataframe
            if df_idx == 0:
                # Start
                ax.scatter(
                    subset["x"].iloc[0],
                    subset["y"].iloc[0],
                    c="#2ecc71",
                    edgecolors="black",
                    s=60,
                    zorder=5,
                )
                # End
                ax.scatter(
                    subset["x"].iloc[-1],
                    subset["y"].iloc[-1],
                    marker="X",
                    c="#e74c3c",
                    edgecolors="black",
                    s=60,
                    zorder=5,
                )

    ax.set_title("Overlaid Trajectory Comparison", fontsize=14, fontweight="bold")
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.grid(visible=True, linestyle="--", alpha=0.3)

    # Legend handling: Show method labels
    ax.legend(title="Interpolation Method", loc="best", frameon=True)

    plt.tight_layout()
    plt.show()


def _generate_test_vel() -> pl.DataFrame:
    t = np.linspace(0, 2 * np.pi, 50)
    dt = t[1] - t[0]  # The 'real' time step

    x = np.sin(2 * t)
    y = np.cos(3 * t)

    # Scale derivatives by dt so they represent change PER FRAME
    dx = 2 * np.cos(2 * t) * dt
    dy = -3 * np.sin(3 * t) * dt

    return pl.DataFrame({
        "frame": np.arange(len(t)),
        "x": x,
        "y": y,
        "vx": dx,
        "vy": dy,
        "track_id": 1,
    }).cast({
        "x": pl.Float64,
        "y": pl.Float64,
        "vx": pl.Float64,
        "vy": pl.Float64,
        "frame": pl.Int64,
        "track_id": pl.Int64,
    })


def _generate_simple() -> pl.DataFrame:
    return pl.DataFrame({
        "frame": np.arange(2),
        "x": [0.0, 1.0],
        "y": [0.0, 1.0],
        "track_id": 1,
    }).cast({
        "x": pl.Float64,
        "y": pl.Float64,
        "frame": pl.Int64,
        "track_id": pl.Int64,
    })


if __name__ == "__main__":
    # 1. Generate the complex data
    df_complex = _generate_simple()

    # 3. Resample (dt=0.2, org_dt=1.0 means 5x upsampling)
    dt_new = 0.4
    dt_org = 1.0
    print("Ratio: ", dt_new / dt_org)
    df_resampled = resample_tracks(
        df_complex,
        ratio=dt_new / dt_org,
        group_by="track_id",
        pos_columns=["x", "y"],
        # vel_columns=["vx", "vy"],
        add_velocity=True,
    )
    print(df_resampled)

    plot_comparison(df_complex, df_resampled)
