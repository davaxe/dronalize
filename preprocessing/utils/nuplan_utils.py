from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


def get_nuplan_scenes_as_pandas(database_path: Path) -> list[pd.DataFrame]:
    """Get NuPlan scenes as a list of pandas DataFrames.

    Args:
        database_path: Path to the NuPlan database file.

    Returns:
        A list of pandas DataFrames, each representing a NuPlan scene.

    """
    return list(get_nuplan_scenes_as_pandas_iter(database_path))


def get_nuplan_scenes_as_pandas_iter(database_path: Path) -> Iterator[pd.DataFrame]:
    """Lazily get NuPlan scenes as pandas DataFrames.

    Args:
        database_path: Path to the NuPlan database file.

    Yields:
        A pandas DataFrame representing a NuPlan scene.

    """
    # Read-only, shared cache, immutable (good for SQLite perf on static DB files)
    conn = sqlite3.connect(
        f"file:{database_path}?mode=ro&immutable=1&cache=shared",
        uri=True,
    )
    # Harmless for read-only workloads; keeps us honest.
    conn.execute("PRAGMA query_only = 1")

    for scene in _SceneHeader.from_db(conn):
        yield _scene_to_dataframe(conn, scene)

    conn.close()


# --- Minimal scene header (joined with log to get location once per scene) ---
_SCENES_QUERY = """
    SELECT s.token AS scene_token, s.name, s.log_token AS log_token, l.location
    FROM scene AS s
    JOIN log   AS l ON l.token = s.log_token
"""


# Pull all lidar boxes for a scene in timestamp order (groups by lidar_pc)
_SCENE_BOXES_QUERY = """
    SELECT
        lp.token     AS lidar_pc_token_hex,
        lp.timestamp      AS lidar_timestamp,
        lb.track_token AS track_token,
        lb.x, lb.y, lb.vx, lb.vy, lb.confidence,
        c.name
    FROM lidar_pc AS lp
    JOIN lidar_box AS lb ON lb.lidar_pc_token = lp.token
    JOIN track AS t ON t.token = lb.track_token
    JOIN category AS c ON c.token = t.category_token
    WHERE lp.scene_token = ?
    ORDER BY lp.timestamp ASC, lp.token ASC, lb.token ASC
"""

_EGO_POSE_QUERY = """
    SELECT
        ep.x, ep.y,
        ep.qw, ep.qx, ep.qy, ep.qz,
        ep.vx, ep.vy,
        ep.acceleration_x, ep.acceleration_y
    FROM lidar_pc AS lp
    JOIN ego_pose AS ep ON ep.token = lp.ego_pose_token
    WHERE lp.scene_token = ?
    ORDER BY lp.timestamp;
"""


@dataclass(frozen=True, slots=True)
class _SceneHeader:
    token: str
    name: str
    log_token: str
    location: str

    @classmethod
    def from_db(cls, conn: sqlite3.Connection) -> Iterator[_SceneHeader]:
        yield from (cls(*row) for row in conn.cursor().execute(_SCENES_QUERY))


class _Frame(TypedDict):
    frame: int
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    ax: float | None
    ay: float | None
    category_name: str
    scene_name: str
    map: str


def _scene_to_dataframe(
    conn: sqlite3.Connection,
    scene: _SceneHeader,
) -> pd.DataFrame:
    cur = conn.cursor()
    ego_cur = conn.cursor()
    ego_cur.execute(_EGO_POSE_QUERY, (scene.token,))

    last_pc: bytes | None = None
    frame_idx = -1
    track_id_map: dict[bytes, int] = {b"ego": 0}

    rows: list[_Frame] = []
    for row in cur.execute(_SCENE_BOXES_QUERY, (scene.token,)):
        lidar_pc, _ts, track, x, y, vx, vy, confidence, category = row

        if lidar_pc != last_pc:
            frame_idx += 1
            last_pc = lidar_pc
            x, y, *_, vx, vy, ax, ay = ego_cur.fetchone()
            rows.append({
                "frame": frame_idx,
                "track_id": 0,
                "x": x,
                "y": y,
                "vx": vx,
                "vy": vy,
                "ax": ax,
                "ay": ay,
                "category_name": "ego-vehicle",
                "scene_name": scene.name,
                "map": scene.location,
            })

        tid = track_id_map.get(track)
        if tid is None:
            tid = len(track_id_map)
            track_id_map[track] = tid

        rows.append({
            "frame": frame_idx,
            "track_id": tid,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "ax": None,
            "ay": None,
            "category_name": category,
            "scene_name": scene.name,
            "map": scene.location,
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    db_path = "data/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"
    for scene_df in get_nuplan_scenes_as_pandas_iter(Path(db_path)):
        print(scene_df.head(1), scene_df.shape, end="\n\n")
