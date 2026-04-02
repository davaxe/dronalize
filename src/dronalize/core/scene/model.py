"""Scene containers and schema-conversion helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.core.errors import SceneSchemaError
from dronalize.core.maps.graph import MapGraph
from dronalize.core.scene.derivations import (
    ConversionContext,
    apply_derivation_plan,
    plan_derivations,
)
from dronalize.core.scene.schema import CANONICAL, SceneField, SceneSchema

if TYPE_CHECKING:
    from dronalize.core.categories import DatasetSplit


MapKey = str | None
"""Stable identifier for the map associated with a scene."""
MapResolver = Callable[["Scene"], MapGraph | None]
"""Callable that materializes a map graph for a scene when needed."""


@dataclass(slots=True, frozen=True)
class Scene:
    """Canonical scene wrapper plus schema metadata."""

    frame: pl.DataFrame
    """DataFrame containing the scene data."""
    scene_number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""
    schema: SceneSchema
    """Schema describing which fields this scene currently provides."""
    sample_time: float | None = None
    """Time interval between frames in seconds."""
    map_key: MapKey = None
    """Stable map identifier for the scene, if one exists."""
    map_resolver: MapResolver | None = field(default=None, compare=False, repr=False)
    """Resolver attached by the loader to materialize the scene map on demand."""
    split_assignment: DatasetSplit | None = None
    """Split assignment for this scene (train/val/test)."""

    @classmethod
    def create(
        cls,
        frame: pl.DataFrame,
        scene_number: int,
        *,
        input_len: int,
        output_len: int,
        schema: SceneSchema,
        sample_time: float | None = None,
        map_key: MapKey = None,
        map_resolver: MapResolver | None = None,
        split_assignment: DatasetSplit | None = None,
        cast_schema: bool = True,
    ) -> Scene:
        """Create a scene with optional schema casting."""
        if cast_schema:
            frame = _cast_to_schema(frame, schema.physical)

        return cls(
            frame=frame,
            scene_number=scene_number,
            input_len=input_len,
            output_len=output_len,
            schema=schema,
            sample_time=sample_time,
            map_key=map_key,
            map_resolver=map_resolver,
            split_assignment=split_assignment,
        )

    def __post_init__(self) -> None:
        """Cast and validate the scene schema."""
        frame = _cast_to_schema(self.frame, self.schema.physical)
        object.__setattr__(self, "frame", frame)

        # Verify schema compatibility with the frame DataFrame
        if not _matches_physical_schema(self.frame.schema, self.schema.physical):
            msg = _get_schema_mismatch_message(
                actual=self.frame.schema,
                expected=self.schema.physical,
                schema_name=self.schema.name,
            )
            raise SceneSchemaError(msg)

    def resolve_map(self) -> MapGraph | None:
        """Materialize the map graph associated with this scene, if available."""
        if self.map_resolver is None:
            return None
        return self.map_resolver(self)

    def has_map(self) -> bool:
        """Return whether this scene can materialize a map graph."""
        return self.map_resolver is not None

    def as_schema(self, schema: SceneSchema = CANONICAL) -> Scene:
        """Return a copy converted to the requested scene schema."""
        if self.schema == schema and _matches_physical_schema(self.frame.schema, schema.physical):
            return self

        return convert_scene(self, schema)

    def with_split_assignment(self, split_assignment: DatasetSplit | None) -> Scene:
        """Return a copy with the given split assignment."""
        return replace(self, split_assignment=split_assignment)

    @override
    def __repr__(self) -> str:
        """Return a compact representation with metadata and DataFrame shape."""
        rows, cols = self.frame.shape
        return (
            "Scene("
            f"scene_number={self.scene_number}, "
            f"input_len={self.input_len}, "
            f"output_len={self.output_len}, "
            f"schema={self.schema.name}, "
            f"map_key={self.map_key!r}, "
            f"frame=DataFrame({rows} rows x {cols} cols)"
            ")"
        )


def convert_scene(scene: Scene, target: SceneSchema = CANONICAL) -> Scene:
    """Convert a scene to the requested schema."""
    return replace(
        scene,
        frame=_convert_frame(
            scene.frame, source=scene.schema, target=target, sample_time=scene.sample_time
        ),
        schema=target,
    )


def derived_scene_fields(
    source: SceneSchema, target: SceneSchema, *, sample_time: float | None
) -> tuple[SceneField, ...]:
    """Return target schema fields that would be materialized by conversion.

    The result is ordered according to the target schema. A `SceneSchemaError` is
    raised when the requested conversion cannot be planned with the available
    source fields and sampling metadata.

    """
    missing_fields = target.fields & ~source.fields
    if not missing_fields:
        return ()

    context = ConversionContext(sample_time=sample_time)
    plan = plan_derivations(source.fields, target.fields, context)
    if plan is None:
        msg = f"No derivation plan found to convert from schema {source.name} to {target.name} "
        raise SceneSchemaError(msg)

    return tuple(field for field in target.ordered_fields() if (missing_fields & field) == field)


def _convert_frame(
    data: pl.DataFrame, *, source: SceneSchema, target: SceneSchema, sample_time: float | None
) -> pl.DataFrame:
    """Convert a data frame from one scene schema to another."""
    semantic = _to_semantic_frame(data, source).sort(["id", "frame"])
    semantic = _derive_missing_fields(
        semantic, source=source, target=target, sample_time=sample_time
    )
    return semantic.select([pl.col(field).cast(dtype) for field, dtype in target.field_items()])


def _to_semantic_frame(data: pl.DataFrame, schema: SceneSchema) -> pl.DataFrame:
    return data.select(schema.semantic_fields())


def _derive_missing_fields(
    data: pl.DataFrame, *, source: SceneSchema, target: SceneSchema, sample_time: float | None
) -> pl.DataFrame:
    if (source.fields & target.fields) == target.fields:
        return data

    context = ConversionContext(sample_time=sample_time)
    plan = plan_derivations(source.fields, target.fields, context)
    if plan is None:
        msg = f"No derivation plan found to convert from schema {source.name} to {target.name} "
        raise SceneSchemaError(msg)

    output, output_fields = apply_derivation_plan(data, plan, context, source.fields)
    if output_fields != target.fields:
        missing = ", ".join(field.to_str() for field in (target.fields & ~output_fields).fields())
        msg = f"Cannot materialize scene schema {target.name}; missing {missing}."
        raise SceneSchemaError(msg)
    return output


def _cast_to_schema(data: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Cast the columns of a DataFrame to match the provided schema.

    This does not remove extra columns or add missing columns; it only casts
    existing columns to the specified types.

    """
    if _matches_physical_schema(data.schema, schema):
        return data
    casts = [pl.col(col).cast(dtype) for col, dtype in schema.items() if data.schema[col] != dtype]
    return data if not casts else data.with_columns(casts)


def _matches_physical_schema(actual: pl.Schema, expected: pl.Schema) -> bool:
    return all(column in actual and actual[column] == dtype for column, dtype in expected.items())


def _get_schema_mismatch_message(
    actual: pl.Schema, expected: pl.Schema, schema_name: str | None = None
) -> str:
    missing = [col for col in expected if col not in actual]
    mismatched = {
        col: f"expected {expected[col]}, got {actual[col]}"
        for col in expected
        if col in actual and actual[col] != expected[col]
    }

    errors: list[str] = []
    if missing:
        errors.append(f"Missing fields: {missing}")
    if mismatched:
        errors.append(f"Mismatched types: {mismatched}")

    if not errors:
        return ""

    prefix = f"Schema mismatch for {schema_name}" if schema_name else "Schema mismatch"
    return f"{prefix}: {'; '.join(errors)}"
