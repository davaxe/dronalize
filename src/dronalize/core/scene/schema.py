"""Canonical scene fields, schemas, and schema lookup helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence  # noqa: TC003
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Final, TypedDict

import polars as pl

from dronalize.core.errors import SceneSchemaError


class SceneField(IntFlag):
    """Canonical semantic fields supported by scene schemas.

    The order of these fields defines the canonical column order for physical
    dataframes, and the presence of a field in a SceneSchema is determined by
    bitwise combination. The `to_str` method returns the canonical physical
    column name for a given field.

    """

    FRAME = auto()
    ID = auto()
    X = auto()
    Y = auto()
    VX = auto()
    VY = auto()
    AX = auto()
    AY = auto()
    YAW = auto()
    AGENT_CATEGORY = auto()

    @classmethod
    def check(cls, value: str) -> bool:
        """Return True if the value is a valid scene field identifier."""
        return value.upper() in cls.__members__

    @classmethod
    def from_str(cls, value: str) -> SceneField:
        """Resolve a canonical string identifier to a SceneField member."""
        try:
            return cls[value.upper()]
        except KeyError as exc:
            msg = f"Invalid SceneField: {value}"
            raise SceneSchemaError(msg) from exc

    def fields(self) -> Iterable[SceneField]:
        """Yield individual SceneField members present in this combination."""
        return (field for field in SceneField if _contains_fields(self, field))

    def to_str(self) -> str:
        """Return the canonical physical column name for this SceneField."""
        if self.name is None:
            msg = f"Field combinations cannot be converted to strings: {self}"
            raise SceneSchemaError(msg)
        return self.name.lower()


_FIELD_DTYPES: Final[dict[SceneField, pl.DataType]] = {
    SceneField.FRAME: pl.Int32(),
    SceneField.ID: pl.Int32(),
    SceneField.X: pl.Float64(),
    SceneField.Y: pl.Float64(),
    SceneField.VX: pl.Float64(),
    SceneField.VY: pl.Float64(),
    SceneField.AX: pl.Float64(),
    SceneField.AY: pl.Float64(),
    SceneField.YAW: pl.Float64(),
    SceneField.AGENT_CATEGORY: pl.Int32(),
}
_FEATURE_FIELDS: Final[SceneField] = (
    SceneField.X
    | SceneField.Y
    | SceneField.VX
    | SceneField.VY
    | SceneField.AX
    | SceneField.AY
    | SceneField.YAW
)


_REQUIRED_FIELDS: Final[SceneField] = (
    SceneField.FRAME | SceneField.ID | SceneField.X | SceneField.Y | SceneField.AGENT_CATEGORY
)


@dataclass(slots=True, frozen=True)
class SceneSchema:
    """Logical scene schema defined as a set of canonical semantic fields."""

    name: str
    fields: SceneField

    def __post_init__(self) -> None:
        """Validate the scene schema."""
        resolved_fields = _resolve_fields(self.fields)
        missing_required = _REQUIRED_FIELDS & ~resolved_fields
        if missing_required:
            missing: str = ", ".join(field.to_str() for field in missing_required.fields())
            msg = f"Scene schemas must include the base fields: {missing}."
            raise SceneSchemaError(msg)

    @classmethod
    def define(cls, name: str, *, fields: SceneField | Iterable[SceneField | str]) -> SceneSchema:
        """Construct a schema from semantic field identifiers."""
        fields = _resolve_fields(fields)
        return cls(name=name, fields=fields)

    @property
    def physical(self) -> pl.Schema:
        """Return the canonical physical dataframe schema for this field set."""
        return pl.Schema({field.to_str(): _FIELD_DTYPES[field] for field in self.ordered_fields()})

    def ordered_fields(self) -> tuple[SceneField, ...]:
        """Return semantic fields in canonical dataframe order."""
        return tuple(self.fields.fields())

    def has(self, *fields: SceneField | str) -> bool:
        """Return whether all requested semantic fields are present."""
        return _contains_fields(self.fields, _resolve_fields(fields))

    def dtype_for(self, field: SceneField | str) -> pl.DataType | None:
        """Return the canonical dtype for a semantic field if present."""
        scene_field = _as_scene_field(field)
        if not _contains_fields(self.fields, scene_field):
            return None
        return _FIELD_DTYPES[scene_field]

    def column_for(self, field: SceneField | str) -> str | None:
        """Return the canonical physical column name for a semantic field."""
        scene_field = _as_scene_field(field)
        if not _contains_fields(self.fields, scene_field):
            return None
        return scene_field.to_str()

    def semantic_fields(self) -> tuple[str, ...]:
        """Return the semantic fields physically present in this schema."""
        return tuple(field.to_str() for field in self.ordered_fields())

    def feature_fields(self) -> tuple[SceneField, ...]:
        """Return the per-agent tensor feature fields in canonical order."""
        return tuple(
            field for field in self.ordered_fields() if _contains_fields(_FEATURE_FIELDS, field)
        )

    def feature_columns(self) -> tuple[str, ...]:
        """Return the per-agent tensor feature column names in canonical order."""
        return tuple(field.to_str() for field in self.feature_fields())

    @property
    def feature_dim(self) -> int:
        """Return the number of per-agent tensor features."""
        return len(self.feature_fields())

    def field_items(self) -> tuple[tuple[str, pl.DataType], ...]:
        """Return the field-name and dtype tuples."""
        return tuple((field.to_str(), _FIELD_DTYPES[field]) for field in self.ordered_fields())


def _as_scene_field(field: SceneField | str) -> SceneField:
    return field if isinstance(field, SceneField) else SceneField.from_str(field)


def _resolve_fields(fields: SceneField | Iterable[SceneField | str]) -> SceneField:
    if isinstance(fields, SceneField):
        return fields

    resolved = SceneField(0)
    for field in fields:
        resolved |= _as_scene_field(field)
    return resolved


def _contains_fields(fields: SceneField, required: SceneField) -> bool:
    return (fields & required) == required


_BASE_FIELDS: Final[SceneField] = (
    SceneField.FRAME | SceneField.ID | SceneField.AGENT_CATEGORY | SceneField.X | SceneField.Y
)

POSITIONS_ONLY: Final[SceneSchema] = SceneSchema.define(
    "positions_only",
    fields=_BASE_FIELDS,
)
POSITIONS_YAW: Final[SceneSchema] = SceneSchema.define(
    "positions_yaw",
    fields=_BASE_FIELDS | SceneField.YAW,
)
POSITIONS_VELOCITY: Final[SceneSchema] = SceneSchema.define(
    "positions_velocity", fields=_BASE_FIELDS | SceneField.VX | SceneField.VY
)
POSITIONS_VELOCITY_YAW: Final[SceneSchema] = SceneSchema.define(
    "positions_velocity_yaw",
    fields=(_BASE_FIELDS | SceneField.VX | SceneField.VY | SceneField.YAW),
)
POSITIONS_VELOCITY_ACCELERATION: Final[SceneSchema] = SceneSchema.define(
    "positions_velocity_acceleration",
    fields=_BASE_FIELDS | SceneField.VX | SceneField.VY | SceneField.AX | SceneField.AY,
)
CANONICAL: Final[SceneSchema] = SceneSchema.define(
    "canonical",
    fields=_BASE_FIELDS
    | SceneField.VX
    | SceneField.VY
    | SceneField.AX
    | SceneField.AY
    | SceneField.YAW,
)

SCENE_SCHEMAS: Final[dict[str, SceneSchema]] = {
    schema.name: schema
    for schema in (
        POSITIONS_ONLY,
        POSITIONS_YAW,
        POSITIONS_VELOCITY,
        POSITIONS_VELOCITY_YAW,
        POSITIONS_VELOCITY_ACCELERATION,
        CANONICAL,
    )
}


def available_scene_schemas() -> tuple[SceneSchema, ...]:
    """Return all registered built-in scene schemas in stable display order."""
    return tuple(SCENE_SCHEMAS.values())


def available_scene_schema_names() -> tuple[str, ...]:
    """Return all registered built-in scene schema names."""
    return tuple(SCENE_SCHEMAS)


class SceneSchemaDefinition(TypedDict):
    """Structured payload used to define a custom scene schema."""

    name: str
    fields: SceneField | Sequence[SceneField | str]


def get_scene_schema(schema: SceneSchema | str | SceneSchemaDefinition) -> SceneSchema:
    """Resolve a schema instance or registered schema name to a SceneSchema."""
    if isinstance(schema, SceneSchema):
        return schema

    if isinstance(schema, dict):
        return SceneSchema.define(schema["name"], fields=schema["fields"])

    if schema in SCENE_SCHEMAS:
        return SCENE_SCHEMAS[schema]

    msg = f"Unknown scene schema '{schema}'."
    raise SceneSchemaError(msg)
