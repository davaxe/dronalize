# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Iterable


def get_opendd_recordings(database_path: Path | str) -> Iterable[pd.DataFrame]:
    """Get OpenDD trajectory recordings from a SQLite database.

    Typically the database file is located at:
    `rdbX/trajectories_rdbX_v3.sqlite`, where `X` is the recording number.

    More information about OpenDD can be found at:
    https://l3pilot.eu/data/opendd.html

    Args:
        database_path: path to the .sqlite database file containing OpenDD
            trajectory recordings.

    Raises:
        FileNotFoundError: If the database file does not exist.


    Yields:
        pd.DataFrame: A DataFrame containing trajectory data for each recording.
            Each DataFrame has columns: frame, track_id, x, y, speed, acc,
            acc_lat, acc_tan, category, recording_id.

    """
    if isinstance(database_path, str):
        database_path = Path(database_path)

    if not database_path.exists():
        msg = f"Database file {database_path} does not exist.   "
        raise FileNotFoundError(msg)

    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = [row[0] for row in cursor.fetchall()]

    for table in table_names:
        if table.startswith("rdb"):
            data_frame = _get_trajectory_recording(cursor, table)
            yield data_frame

    conn.close()


def _get_trajectory_recording(
    cursor: sqlite3.Cursor,
    table: str,
) -> pd.DataFrame:
    """Get trajectory recording from a specific table in the OpenDD database."""
    cursor.execute(f"SELECT * FROM {table}")  # noqa: S608
    data_rows: list[_Frame] = []

    for row in cursor.fetchall():
        data = TrajectoryDataRow(*row)
        data_rows.append(
            _Frame(
                frame=data.timestamp,
                track_id=data.object_id,
                x=data.utm_x,
                y=data.utm_y,
                speed=data.velocity,
                acc=data.acceleration,
                acc_lat=data.acceleration_lateral,
                acc_tan=data.acceleration_tangential,
                category=data.object_category,
                recording=table,
            ),
        )

    recording = pd.DataFrame(data_rows)
    # Convert timestamps in seconds to frame indices
    recording["frame_rounded"] = recording["frame"].round(4)
    unique_sorted = sorted(recording["frame_rounded"].unique())
    timestamp_to_index = {ts: idx for idx, ts in enumerate(unique_sorted)}
    recording["frame"] = recording["frame_rounded"].map(timestamp_to_index)
    return recording.drop(columns="frame_rounded")


@dataclass(frozen=True, slots=True)
class TrajectoryDataRow:
    """Single row of trajectory data."""

    id: int
    timestamp: int
    object_id: int
    utm_x: float
    utm_y: float
    utm_angle: float
    velocity: float
    acceleration: float
    acceleration_lateral: float
    acceleration_tangential: float
    object_category: str
    width: float
    length: float
    trailer_id: int | None = None


class _Frame(TypedDict):
    frame: int
    track_id: int
    x: float
    y: float
    speed: float
    acc: float
    acc_lat: float
    acc_tan: float
    category: str
    recording: str


if __name__ == "__main__":
    # Example usage
    rdb = "rdb1"
    db_path = Path(
        f"../datasets/openDD/opendd_v3-{rdb}/{rdb}/trajectories_{rdb}_v3.sqlite"
    )

    for recording in get_opendd_recordings(db_path):
        print(recording.head())
        break
