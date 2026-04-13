from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from dronalize.core import AgentCategory
from dronalize.core.errors import TrajectorySchemaError
from dronalize.core.maps import MapGraph
from dronalize.core.scene import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_YAW,
    Scene,
    TrajectoryField,
    TrajectorySchema,
)
from dronalize.io.encoding import encode_scene_record


def test_schema_definition_normalizes_column_order() -> None:
    schema = TrajectorySchema.define(
        "custom",
        fields=(
            TrajectoryField.Y,
            TrajectoryField.FRAME,
            TrajectoryField.ID,
            TrajectoryField.X,
            TrajectoryField.VY,
            TrajectoryField.VX,
            TrajectoryField.AGENT_CATEGORY,
        ),
    )

    assert schema.semantic_fields() == ("frame", "id", "x", "y", "vx", "vy", "agent_category")
    assert schema.feature_columns() == ("x", "y", "vx", "vy")


def test_schema_rejects_missing_required_base_fields() -> None:
    with pytest.raises(TrajectorySchemaError, match="base fields"):
        _ = TrajectorySchema.define(
            "invalid", fields=(TrajectoryField.FRAME, TrajectoryField.ID, TrajectoryField.X)
        )


def test_scene_as_schema_derives_kinematics_from_positions() -> None:
    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [AgentCategory.CAR] * 3,
        }),
        scene_number=1,
        history_frames=2,
        future_frames=1,
        schema=POSITIONS_ONLY,
        sample_time=1.0,
    )

    converted = scene.as_schema(CANONICAL)

    assert converted.schema == CANONICAL
    np.testing.assert_allclose(converted.frame["vx"].to_numpy(), np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(converted.frame["ax"].to_numpy(), np.array([0.0, 0.0, 0.0]))


def test_scene_as_schema_without_sample_time_fails_for_kinematics() -> None:
    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [AgentCategory.CAR] * 3,
        }),
        scene_number=1,
        history_frames=2,
        future_frames=1,
        schema=POSITIONS_ONLY,
        sample_time=None,
    )

    with pytest.raises(TrajectorySchemaError, match="No derivation plan"):
        _ = scene.as_schema(CANONICAL)

    yaw_only = scene.as_schema(POSITIONS_YAW)
    assert yaw_only.schema == POSITIONS_YAW


def test_scene_map_resolver_is_deferred_and_usable_in_encoding() -> None:
    graph = MapGraph(
        node_positions=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64),
        edge_indices=np.array([[0], [1]], dtype=np.int32),
    )
    calls = {"count": 0}

    def _resolver(_: Scene) -> MapGraph:
        calls["count"] += 1
        return graph

    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "vx": [1.0, 1.0, 1.0],
            "vy": [0.0, 0.0, 0.0],
            "ax": [0.0, 0.0, 0.0],
            "ay": [0.0, 0.0, 0.0],
            "yaw": [0.0, 0.0, 0.0],
            "agent_category": [AgentCategory.CAR] * 3,
        }),
        scene_number=9,
        history_frames=2,
        future_frames=1,
        schema=CANONICAL,
        sample_time=1.0,
        map_resolver=_resolver,
    )

    assert scene.has_map()
    _ = scene.resolve_map()
    encoded = encode_scene_record(scene, dtype=np.float32)

    assert calls["count"] >= 2
    assert encoded.map_node_positions.shape == (2, 2)
