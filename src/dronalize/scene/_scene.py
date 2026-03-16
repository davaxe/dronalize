from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.maps.graph import MapGraph
from dronalize.scene._derivations import (
    ConversionContext,
    apply_derivation_plan,
    plan_derivations,
    requires_sample_time,
)
from dronalize.scene._schema import CANONICAL_V1, SceneField, SceneSchema

if TYPE_CHECKING:
    from dronalize.categories import DatasetSplit


MapKey = str | None
"""Lightweight map identifier stored on each scene."""


@dataclass(slots=True, frozen=True)
class Scene:
    """Canonical scene wrapper plus schema metadata."""

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""
    schema: SceneSchema
    """Schema describing the current `inner` dataframe layout."""
    sample_time: float | None = None
    """Time interval between frames in seconds."""
    map_key: MapKey = None
    """Lightweight map identifier for the scene."""
    map_resolver: MapResolver | None = field(default=None, compare=False, repr=False)
    """Resolver attached by the loader that produced this scene."""
    split_assignment: DatasetSplit | None = None
    """Split assignment for this scene (train/val/test)."""

    @classmethod
    def create(
        cls,
        inner: pl.DataFrame,
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
        if cast_schema:
            inner = _cast_to_schema(inner, schema.physical)

        return cls(
            inner=inner,
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
        inner = _cast_to_schema(self.inner, self.schema.physical)
        if inner is not self.inner:
            object.__setattr__(self, "inner", inner)

        # Verify schema compatibility with the inner DataFrame
        if not _matches_physical_schema(self.inner.schema, self.schema.physical):
            msg = _get_schema_mismatch_message(
                actual=self.inner.schema,
                expected=self.schema.physical,
                schema_name=self.schema.name,
            )
            raise ValueError(msg)

    def resolve_map(self) -> MapGraph | None:
        """Resolve this scene's `map_key` into a `MapGraph`."""
        if self.map_resolver is None:
            return None
        return self.map_resolver(self)

    def has_map(self) -> bool:
        """Check if this scene has an attached map resolver and key."""
        return self.map_resolver is not None

    def as_schema(self, schema: SceneSchema = CANONICAL_V1) -> Scene:
        """Return a copy converted to the requested scene schema."""
        if self.schema == schema and _matches_physical_schema(self.inner.schema, schema.physical):
            return self

        return convert_scene(self, schema)

    def override_split_assignment(self, split_assignment: DatasetSplit | None) -> Scene:
        """Override the split assignment for this scene."""
        return replace(self, split_assignment=split_assignment)

    @override
    def __repr__(self) -> str:
        """Return a compact representation with metadata and DataFrame shape."""
        rows, cols = self.inner.shape
        return (
            f"scene_number={self.number}, "
            f"input_len={self.input_len}, "
            f"output_len={self.output_len}, "
            f"schema={self.schema.name}/v{self.schema.version}, "
            f"map_key={self.map_key!r}, "
            f"inner=DataFrame({rows} rows x {cols} cols))"
        )


MapResolver = Callable[[Scene], MapGraph | None]
"""Protocol for resolving a `MapKey` into a `MapGraph` for a given `Scene`."""


def convert_scene(scene: Scene, target: SceneSchema = CANONICAL_V1) -> Scene:
    """Convert a scene to the requested schema."""
    return replace(
        scene,
        inner=convert_frame(
            scene.inner,
            source=scene.schema,
            target=target,
            sample_time=scene.sample_time,
        ),
        schema=target,
    )


def convert_frame(
    data: pl.DataFrame,
    *,
    source: SceneSchema,
    target: SceneSchema,
    sample_time: float | None,
) -> pl.DataFrame:
    """Convert a data frame from one scene schema to another."""
    semantic = _to_semantic_frame(data, source).sort(["id", "frame"])
    semantic = _derive_missing_fields(
        semantic,
        source=source,
        target=target,
        sample_time=sample_time,
    )
    return semantic.select([
        pl.col(field).cast(dtype).alias(column) for field, column, dtype in target.field_items()
    ])


def _to_semantic_frame(data: pl.DataFrame, schema: SceneSchema) -> pl.DataFrame:
    return data.select([pl.col(column).alias(field) for field, column, _ in schema.field_items()])


def _derive_missing_fields(
    data: pl.DataFrame,
    *,
    source: SceneSchema,
    target: SceneSchema,
    sample_time: float | None,
) -> pl.DataFrame:
    available_fields = source.fields
    required_fields = target.fields
    if (available_fields & required_fields) == required_fields:
        return data

    context = ConversionContext(sample_time=sample_time)
    plan = plan_derivations(source.ordered_fields(), target.ordered_fields(), context)
    if plan is None:
        missing_fields = required_fields & ~available_fields
        if requires_sample_time(missing_fields.fields(), available_fields.fields(), context):
            msg = "Scene schema conversion requires sample_time to derive kinematics."
            raise ValueError(msg)
        missing = ", ".join(field.to_str() for field in missing_fields.fields())
        msg = f"Cannot materialize scene schema {target.name}/v{target.version}; missing {missing}."
        raise ValueError(msg)

    output = apply_derivation_plan(data.lazy(), plan, context).collect()
    output_fields = SceneField(0)
    for column in output.columns:
        if SceneField.check(column):
            output_fields |= SceneField.from_str(column)

    if (output_fields & required_fields) != required_fields:
        missing = ", ".join(field.to_str() for field in (required_fields & ~output_fields).fields())
        msg = f"Cannot materialize scene schema {target.name}/v{target.version}; missing {missing}."
        raise ValueError(msg)
    return output


def _cast_to_schema(data: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    if not _matches_physical_schema_name(data.schema, schema):
        return data

    casts = [pl.col(col).cast(dtype) for col, dtype in schema.items() if data.schema[col] != dtype]
    return data if not casts else data.with_columns(casts)


def _matches_physical_schema(actual: pl.Schema, expected: pl.Schema) -> bool:
    return all(column in actual and actual[column] == dtype for column, dtype in expected.items())


def _matches_physical_schema_name(actual: pl.Schema, expected: pl.Schema) -> bool:
    """Check if the actual schema matches the expected schema, ignoring dtypes."""
    return all(column in actual for column in expected)


def _get_schema_mismatch_message(
    actual: pl.Schema,
    expected: pl.Schema,
    schema_name: str | None = None,
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

    return f"Schema mismatch for {schema_name or ''} - {'; '.join(errors)}"
