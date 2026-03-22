from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

import polars as pl
from typing_extensions import override

from dronalize.maps.graph import MapGraph
from dronalize.scene._derivations import ConversionContext, apply_derivation_plan, plan_derivations
from dronalize.scene._schema import CANONICAL_V1, SceneField, SceneSchema

if TYPE_CHECKING:
    from dronalize.categories import DatasetSplit


MapKey = str | None
"""Lightweight map identifier stored on each scene."""
MapResolver = Callable[["Scene"], MapGraph | None]


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
            number=scene_number,
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
        """Check if this scene has an attached map resolver."""
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
            "Scene("
            f"number={self.number}, "
            f"input_len={self.input_len}, "
            f"output_len={self.output_len}, "
            f"schema={self.schema.name}, "
            f"map_key={self.map_key!r}, "
            f"inner=DataFrame({rows} rows x {cols} cols)"
            ")"
        )


def convert_scene(scene: Scene, target: SceneSchema = CANONICAL_V1) -> Scene:
    """Convert a scene to the requested schema."""
    return replace(
        scene,
        inner=_convert_frame(
            scene.inner,
            source=scene.schema,
            target=target,
            sample_time=scene.sample_time,
        ),
        schema=target,
    )


def derived_scene_fields(
    source: SceneSchema,
    target: SceneSchema,
    *,
    sample_time: float | None,
) -> tuple[SceneField, ...]:
    """Return target schema fields that would be materialized by conversion.

    The result is ordered according to the target schema. A `ValueError` is
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
        raise ValueError(msg)

    return tuple(field for field in target.ordered_fields() if (missing_fields & field) == field)


def _convert_frame(
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
    return semantic.select([pl.col(field).cast(dtype) for field, dtype in target.field_items()])


def _to_semantic_frame(data: pl.DataFrame, schema: SceneSchema) -> pl.DataFrame:
    return data.select(schema.semantic_fields())


def _derive_missing_fields(
    data: pl.DataFrame,
    *,
    source: SceneSchema,
    target: SceneSchema,
    sample_time: float | None,
) -> pl.DataFrame:
    if (source.fields & target.fields) == target.fields:
        return data

    context = ConversionContext(sample_time=sample_time)
    plan = plan_derivations(source.fields, target.fields, context)
    if plan is None:
        msg = f"No derivation plan found to convert from schema {source.name} to {target.name} "
        raise ValueError(msg)

    output, output_fields = apply_derivation_plan(data, plan, context, source.fields)
    if output_fields != target.fields:
        missing = ", ".join(field.to_str() for field in (target.fields & ~output_fields).fields())
        msg = f"Cannot materialize scene schema {target.name}; missing {missing}."
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

    prefix = f"Schema mismatch for {schema_name}" if schema_name else "Schema mismatch"
    return f"{prefix}: {'; '.join(errors)}"
