from __future__ import annotations

import binascii
import sqlite3
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterator


@dataclass(frozen=True, slots=True)
class SceneDescription:
    token: bytes
    name: str
    log_token: bytes
    start_lidar_pc_token: bytes
    end_lidar_pc_token: bytes
    num_lidar_pcs: int

    @classmethod
    def from_db(cls, conn: sqlite3.Connection) -> Iterator[SceneDescription]:
        cursor = conn.cursor()
        for row in cursor.execute(SCENES_QUERY):
            yield cls(*row)


@dataclass(frozen=True, slots=True)
class LidarPC:
    token: bytes
    next_token: bytes
    prev_token: bytes
    ego_pose_token: bytes
    timestamp: int

    @classmethod
    def from_db_by_token(
        cls,
        conn: sqlite3.Connection,
        lidar_pc_token: bytes,
    ) -> LidarPC:
        query = f"""
            SELECT lp.token, lp.next_token, lp.prev_token, lp.ego_pose_token, lp.timestamp
            FROM lidar_pc lp
            WHERE lp.token = x'{binascii.hexlify(lidar_pc_token).decode()}'
        """  # noqa: S608
        cursor = conn.cursor()
        row = cursor.execute(query).fetchone()
        return cls(*row)


@dataclass(frozen=True, slots=True)
class EgoPose:
    token: bytes
    x: float
    y: float
    qw: float
    qx: float
    qy: float
    qz: float
    vx: float
    vy: float
    acc_x: float
    acc_y: float
    epsg: int
    timestamp: int

    @classmethod
    def from_db_by_token(
        cls,
        conn: sqlite3.Connection,
        ego_pose_token: bytes,
    ) -> EgoPose:
        query = f"""
            SELECT ep.token, ep.x, ep.y, ep.qw, ep.qx, ep.qy, ep.qz, ep.vx, ep.vy,
            ep.acceleration_x, ep.acceleration_y, ep.epsg, ep.timestamp
            FROM ego_pose ep
            WHERE ep.token = x'{binascii.hexlify(ego_pose_token).decode()}'
        """  # noqa: S608
        cursor = conn.cursor()
        row = cursor.execute(query).fetchone()
        return cls(*row)


@dataclass(frozen=True, slots=True)
class LidarBox:
    token: bytes
    track_token: bytes
    next_token: bytes
    prev_token: bytes
    x: float
    y: float
    vx: float
    vy: float
    confidence: float

    @classmethod
    def from_db_by_token(
        cls,
        conn: sqlite3.Connection,
        lidar_pc_token: bytes,
    ) -> Iterator[LidarBox]:
        query = f"""
            SELECT lb.token, lb.track_token, lb.next_token, lb.prev_token, lb.x, lb.y,
                   lb.vx, lb.vy, lb.confidence
            FROM lidar_box lb
            WHERE lb.lidar_pc_token = x'{binascii.hexlify(lidar_pc_token).decode()}'
        """  # noqa: S608
        cursor = conn.cursor()
        for row in cursor.execute(query):
            yield cls(*row)


@dataclass(frozen=True, slots=True)
class Track:
    token: bytes
    category: str
    category_description: str

    @classmethod
    def from_db_by_token(
        cls,
        conn: sqlite3.Connection,
        track_token: bytes,
    ) -> Track:
        query = f"""
            SELECT t.token, t.category_token
            FROM track t
            WHERE t.token = x'{binascii.hexlify(track_token).decode()}'
        """  # noqa: S608
        cursor = conn.cursor()
        track_token, category_token = cursor.execute(query).fetchone()
        category_name, desc = get_category(conn, category_token)
        return cls(track_token, category_name, desc)


@lru_cache(maxsize=128)
def get_category(conn: sqlite3.Connection, category_token: bytes) -> tuple[str, str]:
    category_query: str = f"""
            SELECT c.name, c.description
            FROM category c
            WHERE c.token = x'{binascii.hexlify(category_token).decode()}'
        """  # noqa: S608
    category_name, category_desc = conn.cursor().execute(category_query).fetchone()
    return category_name, category_desc


def get_map_location(conn: sqlite3.Connection, log_token: bytes) -> str:
    query = f"""
        SELECT l.location
        FROM log l
        WHERE l.token = x'{binascii.hexlify(log_token).decode()}'
    """  # noqa: S608
    cursor = conn.cursor()
    location, *_ = cursor.execute(query).fetchone()
    return location


SCENES_QUERY = """
    SELECT s.token AS scene_token, s.name, s.log_token,
           (SELECT lp.token
            FROM lidar_pc lp
            WHERE lp.scene_token = s.token
            ORDER BY lp.timestamp ASC
            LIMIT 1) AS start_lidarpc_token,
           (SELECT lp.token
            FROM lidar_pc lp
            WHERE lp.scene_token = s.token
            ORDER BY lp.timestamp DESC
            LIMIT 1) AS end_lidarpc_token,
           COUNT(lp_all.token) AS num_lidar_pcs
    FROM scene s
    JOIN lidar_pc lp_all ON lp_all.scene_token = s.token
    GROUP BY s.token
"""


class _Frame(TypedDict):
    frame: int
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    scene_name: str
    map: str


def get_nuplan_scenes_as_pandas(database_path: Path) -> list[pd.DataFrame]:
    return list(get_nuplan_scenes_as_pandas_iter(database_path))


def get_nuplan_scenes_as_pandas_iter(database_path: Path) -> Iterator[pd.DataFrame]:
    conn = sqlite3.connect(
        f"file:{database_path}?mode=ro&immutable=1&cache=shared",
        uri=True,
    )
    for scene_name in SceneDescription.from_db(conn):
        yield pd.DataFrame(_get_scene_frames(conn, scene_name))


def _get_scene_frames(
    conn: sqlite3.Connection,
    scene_name: SceneDescription,
) -> list[_Frame]:
    current: LidarPC | None = LidarPC.from_db_by_token(
        conn,
        scene_name.start_lidar_pc_token,
    )
    map_location = get_map_location(conn, scene_name.log_token)
    tracks: dict[bytes, int] = {}

    frames: list[_Frame] = []
    frame_counter = 0
    while current:
        for lidar_box in LidarBox.from_db_by_token(conn, current.token):
            track = Track.from_db_by_token(conn, lidar_box.track_token)
            track_id = tracks.get(lidar_box.track_token)
            if track_id is None:
                track_id = len(tracks) + 1
                tracks[lidar_box.track_token] = track_id

            frames.append({
                "frame": frame_counter,
                "track_id": track_id,
                "x": lidar_box.x,
                "y": lidar_box.y,
                "vx": lidar_box.vx,
                "vy": lidar_box.vy,
                "scene_name": scene_name.name,
                "map": map_location,
            })

        if current.next_token is None:
            current = None
            break

        current = LidarPC.from_db_by_token(conn, current.next_token)

    return frames


if __name__ == "__main__":
    db_path = "data/mini/2021.05.12.22.00.38_veh-35_01008_01518.db"  # adjust

    for scene in get_nuplan_scenes_as_pandas_iter(Path(db_path)):
        print(scene.head(), scene.shape)
