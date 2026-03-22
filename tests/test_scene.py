# pyright: standard
# ruff: noqa: PLC2701

import numpy as np
import polars as pl
import pytest

from dronalize.scene import (
    CANONICAL_V1,
    POSITIONS_ONLY_V1,
    POSITIONS_YAW_V1,
    Scene,
    SceneField,
    SceneSchema,
)
from dronalize.scene._derivations import ConversionContext, plan_derivations
from dronalize.storage.encoding import scene_to_numpy_dict


def test_scene_schema_define_uses_semantic_fields_in_canonical_order() -> None:
    """Test that field-based schema definitions normalize to canonical dataframe order."""
    schema = SceneSchema.define(
        name="custom",
        fields=(
            SceneField.Y,
            SceneField.FRAME,
            SceneField.VY,
            SceneField.ID,
            SceneField.VX,
            SceneField.X,
        ),
    )

    assert schema.semantic_fields() == ("frame", "id", "x", "y", "vx", "vy")
    assert schema.column_for(SceneField.VX) == "vx"
    assert schema.dtype_for("vx") == pl.Float64()
    assert schema.has(SceneField.FRAME, "id", "x", "y")


def test_scene_schema_accepts_combined_intflag_fields() -> None:
    """Test that schemas can be defined directly from an IntFlag bitmask."""
    schema = SceneSchema.define(
        "positions_velocity",
        fields=(
            SceneField.FRAME
            | SceneField.ID
            | SceneField.X
            | SceneField.Y
            | SceneField.VX
            | SceneField.VY
        ),
    )

    assert schema.fields == (
        SceneField.FRAME
        | SceneField.ID
        | SceneField.X
        | SceneField.Y
        | SceneField.VX
        | SceneField.VY
    )
    assert schema.semantic_fields() == ("frame", "id", "x", "y", "vx", "vy")


def test_plan_derivations_accepts_scene_field_bitmasks() -> None:
    """Derivation planning should operate directly on SceneField bitmasks."""
    plan = plan_derivations(
        SceneField.X | SceneField.Y,
        SceneField.VX | SceneField.VY | SceneField.YAW,
        ConversionContext(sample_time=1.0),
    )

    assert plan is not None
    assert tuple(rule.name for rule in plan) == ("velocity_from_position", "yaw_from_velocity")


def test_scene_schema_requires_base_fields() -> None:
    """Test that schemas must include the base frame, id, and position fields."""
    with pytest.raises(ValueError, match="base fields"):
        SceneSchema.define(name="invalid", fields=(SceneField.FRAME, SceneField.ID, SceneField.X))


def test_scene_as_schema_derives_velocity_acceleration_and_yaw() -> None:
    """Test that schema conversion derives canonical kinematics from positions."""
    scene = Scene(
        inner=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [0, 0, 0],
        }),
        number=7,
        input_len=2,
        output_len=1,
        schema=POSITIONS_ONLY_V1,
        sample_time=1.0,
    )

    converted = scene.as_schema(CANONICAL_V1)

    assert converted.schema == CANONICAL_V1
    assert converted.inner["vx"].to_list() == pytest.approx([1.0, 1.0, 1.0])
    assert converted.inner["vy"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert converted.inner["ax"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert converted.inner["ay"].to_list() == pytest.approx([0.0, 0.0, 0.0])
    assert converted.inner["yaw"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_scene_as_schema_derives_yaw_from_position_without_sample_time() -> None:
    """Test that yaw-only conversion can use position differences without sample time."""
    scene = Scene(
        inner=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [0, 0, 0],
        }),
        number=9,
        input_len=2,
        output_len=1,
        schema=POSITIONS_ONLY_V1,
    )

    converted = scene.as_schema(POSITIONS_YAW_V1)

    assert converted.schema == POSITIONS_YAW_V1
    assert converted.inner["yaw"].to_list() == pytest.approx([0.0, 0.0, 0.0])


def test_scene_as_schema_requires_sample_time_for_derivatives() -> None:
    """Test that kinematic derivation fails clearly without sample-time metadata."""
    scene = Scene(
        inner=pl.DataFrame({
            "frame": [0, 1],
            "id": [1, 1],
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "agent_category": [0, 0],
        }),
        number=1,
        input_len=1,
        output_len=1,
        schema=POSITIONS_ONLY_V1,
    )

    with pytest.raises(ValueError, match="sample_time"):
        scene.as_schema(CANONICAL_V1)


def test_scene_schema_feature_columns_follow_canonical_tensor_order() -> None:
    """Feature columns should reflect only persisted tensor fields in canonical order."""
    assert POSITIONS_ONLY_V1.feature_columns() == ("x", "y")
    assert POSITIONS_YAW_V1.feature_columns() == ("x", "y", "yaw")
    assert CANONICAL_V1.feature_columns() == ("x", "y", "vx", "vy", "ax", "ay", "yaw")


def test_scene_to_numpy_dict_respects_requested_scene_schema() -> None:
    """NumPy conversion should derive feature width from the requested scene schema."""
    scene = Scene(
        inner=pl.DataFrame({
            "frame": [0, 1, 2],
            "id": [1, 1, 1],
            "x": [0.0, 1.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "agent_category": [0, 0, 0],
        }),
        number=3,
        input_len=2,
        output_len=1,
        schema=POSITIONS_ONLY_V1,
        sample_time=1.0,
    )

    canonical = scene_to_numpy_dict(scene, scene_schema=CANONICAL_V1, dtype=np.float64)
    positions_only = scene_to_numpy_dict(scene, scene_schema=POSITIONS_ONLY_V1, dtype=np.float64)

    assert canonical["input_features"].shape == (1, 2, 7)
    assert canonical["target_features"].shape == (1, 1, 7)
    assert positions_only["input_features"].shape == (1, 2, 2)
    assert positions_only["target_features"].shape == (1, 1, 2)
