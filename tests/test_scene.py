# pyright: standard

import numpy as np
import polars as pl
import pytest

from dronalize.core.errors import TrajectorySchemaError
from dronalize.core.scene import (
    CANONICAL,
    POSITIONS_ONLY,
    POSITIONS_YAW,
    Scene,
    TrajectoryField,
    TrajectorySchema,
)
from dronalize.core.scene.derivations import ConversionContext, plan_derivations
from dronalize.io.encoding import encode_scene_record


def test_trajectory_schema_orders_fields() -> None:
    """Test that field-based schema definitions normalize to canonical dataframe order."""
    schema = TrajectorySchema.define(
        name="custom",
        fields=(
            TrajectoryField.Y,
            TrajectoryField.FRAME,
            TrajectoryField.VY,
            TrajectoryField.ID,
            TrajectoryField.VX,
            TrajectoryField.X,
            TrajectoryField.AGENT_CATEGORY,
        ),
    )

    assert schema.semantic_fields() == ("frame", "id", "x", "y", "vx", "vy", "agent_category")
    assert schema.column_for(TrajectoryField.VX) == "vx"
    assert schema.dtype_for("vx") == pl.Float64()
    assert schema.has(TrajectoryField.FRAME, "id", "x", "y")


def test_trajectory_schema_accepts_bitmasks() -> None:
    """Test that schemas can be defined directly from an IntFlag bitmask."""
    schema = TrajectorySchema.define(
        "positions_velocity",
        fields=(
            TrajectoryField.FRAME
            | TrajectoryField.ID
            | TrajectoryField.X
            | TrajectoryField.Y
            | TrajectoryField.VX
            | TrajectoryField.VY
            | TrajectoryField.AGENT_CATEGORY
        ),
    )

    assert schema.fields == (
        TrajectoryField.FRAME
        | TrajectoryField.ID
        | TrajectoryField.X
        | TrajectoryField.Y
        | TrajectoryField.VX
        | TrajectoryField.VY
        | TrajectoryField.AGENT_CATEGORY
    )
    assert schema.semantic_fields() == ("frame", "id", "x", "y", "vx", "vy", "agent_category")


def test_plan_derivations_bitmasks() -> None:
    """Derivation planning should operate directly on TrajectoryField bitmasks."""
    plan = plan_derivations(
        TrajectoryField.X | TrajectoryField.Y,
        TrajectoryField.VX | TrajectoryField.VY | TrajectoryField.YAW,
        ConversionContext(sample_time=1.0),
    )

    assert plan is not None
    assert tuple(rule.name for rule in plan) == ("velocity_from_position", "yaw_from_velocity")


def test_trajectory_schema_requires_bases() -> None:
    """Test that schemas must include the base frame, id, and position fields."""
    with pytest.raises(TrajectorySchemaError, match="base fields"):
        TrajectorySchema.define(
            name="invalid", fields=(TrajectoryField.FRAME, TrajectoryField.ID, TrajectoryField.X)
        )


def test_scene_derives_kinematics() -> None:
    """Test that schema conversion derives canonical kinematics from positions."""
    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [0, 0, 0],
        }),
        scene_number=7,
        input_len=2,
        output_len=1,
        schema=POSITIONS_ONLY,
        sample_time=1.0,
    )

    converted = scene.as_schema(CANONICAL)

    assert converted.schema == CANONICAL
    assert converted.frame["vx"].to_list() == pytest.approx([1.0, 1.0, 1.0])
    assert converted.frame["vy"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert converted.frame["ax"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert converted.frame["ay"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert converted.frame["yaw"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_scene_derives_yaw_without_sample_time() -> None:
    """Test that yaw-only conversion can use position differences without sample time."""
    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [0, 0, 0],
        }),
        scene_number=9,
        input_len=2,
        output_len=1,
        schema=POSITIONS_ONLY,
    )

    converted = scene.as_schema(POSITIONS_YAW)

    assert converted.schema == POSITIONS_YAW
    assert converted.frame["yaw"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_scene_requires_sample_time() -> None:
    """Test that kinematic derivation fails clearly without sample-time metadata."""
    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1],
            "id": [1, 1],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "agent_category": [0, 0],
        }),
        scene_number=1,
        input_len=1,
        output_len=1,
        schema=POSITIONS_ONLY,
    )

    with pytest.raises(TrajectorySchemaError, match="No derivation plan"):
        scene.as_schema(CANONICAL)


def test_feature_columns_canonical_order() -> None:
    """Feature columns should reflect only persisted tensor fields in canonical order."""
    assert POSITIONS_ONLY.feature_columns() == ("x", "y")
    assert POSITIONS_YAW.feature_columns() == ("x", "y", "yaw")
    assert CANONICAL.feature_columns() == ("x", "y", "vx", "vy", "ax", "ay", "yaw")


def test_encode_scene_record_schema() -> None:
    """NumPy conversion should derive feature width from the requested trajectory schema."""
    scene = Scene(
        frame=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [0, 0, 0],
        }),
        scene_number=3,
        input_len=2,
        output_len=1,
        schema=POSITIONS_ONLY,
        sample_time=1.0,
    )

    canonical = encode_scene_record(scene, trajectory_schema=CANONICAL, dtype=np.float64)
    positions_only = encode_scene_record(scene, trajectory_schema=POSITIONS_ONLY, dtype=np.float64)

    assert canonical["features"].shape == (1, 3, 7)
    assert positions_only["features"].shape == (1, 3, 2)
