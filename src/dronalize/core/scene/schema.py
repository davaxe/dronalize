"""Canonical trajectory fields, schemas, and schema lookup helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence  # noqa: TC003
from dataclasses import dataclass
from enum import IntFlag, auto
from typing import Final

import polars as pl
from typing_extensions import TypedDict

from dronalize.core.errors import TrajectorySchemaError


class TrajectoryField(IntFlag):
    """Canonical semantic fields supported by trajectory schemas.

    The order of these fields defines the canonical column order for physical
    dataframes, and the presence of a field in a TrajectorySchema is determined by
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
        """Return True if the value is a valid trajectory field identifier."""
        return value.upper() in cls.__members__

    @classmethod
    def from_str(cls, value: str) -> TrajectoryField:
        """Resolve a canonical string identifier to a TrajectoryField member."""
        try:
            return cls[value.upper()]
        except KeyError as exc:
            msg = f"Invalid TrajectoryField: {value}"
            raise TrajectorySchemaError(msg) from exc

    def fields(self) -> Iterable[TrajectoryField]:
        """Yield individual TrajectoryField members present in this combination."""
        return (field for field in TrajectoryField if _contains_fields(self, field))

    def to_str(self) -> str:
        """Return the canonical physical column name for this TrajectoryField."""
        if self.name is None:
            msg = f"Field combinations cannot be converted to strings: {self}"
            raise TrajectorySchemaError(msg)
        return self.name.lower()


_FIELD_DTYPES: Final[dict[TrajectoryField, pl.DataType]] = {
    TrajectoryField.FRAME: pl.Int32(),
    TrajectoryField.ID: pl.Int32(),
    TrajectoryField.X: pl.Float64(),
    TrajectoryField.Y: pl.Float64(),
    TrajectoryField.VX: pl.Float64(),
    TrajectoryField.VY: pl.Float64(),
    TrajectoryField.AX: pl.Float64(),
    TrajectoryField.AY: pl.Float64(),
    TrajectoryField.YAW: pl.Float64(),
    TrajectoryField.AGENT_CATEGORY: pl.Int32(),
}
_FEATURE_FIELDS: Final[TrajectoryField] = (
    TrajectoryField.X
    | TrajectoryField.Y
    | TrajectoryField.VX
    | TrajectoryField.VY
    | TrajectoryField.AX
    | TrajectoryField.AY
    | TrajectoryField.YAW
)


_REQUIRED_FIELDS: Final[TrajectoryField] = (
    TrajectoryField.FRAME
    | TrajectoryField.ID
    | TrajectoryField.X
    | TrajectoryField.Y
    | TrajectoryField.AGENT_CATEGORY
)


@dataclass(slots=True, frozen=True)
class TrajectorySchema:
    """Logical trajectory schema defined as a set of canonical semantic fields."""

    name: str
    fields: TrajectoryField

    def __post_init__(self) -> None:
        """Validate the trajectory schema."""
        resolved_fields = _resolve_fields(self.fields)
        missing_required = _REQUIRED_FIELDS & ~resolved_fields
        if missing_required:
            missing: str = ", ".join(field.to_str() for field in missing_required.fields())
            msg = f"Trajectory schemas must include the base fields: {missing}."
            raise TrajectorySchemaError(msg)

    @classmethod
    def define(
        cls, name: str, *, fields: TrajectoryField | Iterable[TrajectoryField | str]
    ) -> TrajectorySchema:
        """Construct a schema from semantic field identifiers."""
        fields = _resolve_fields(fields)
        return cls(name=name, fields=fields)

    @property
    def physical(self) -> pl.Schema:
        """Return the canonical physical dataframe schema for this field set."""
        return pl.Schema({field.to_str(): _FIELD_DTYPES[field] for field in self.ordered_fields()})

    def ordered_fields(self) -> tuple[TrajectoryField, ...]:
        """Return semantic fields in canonical dataframe order."""
        return tuple(self.fields.fields())

    def has(self, *fields: TrajectoryField | str) -> bool:
        """Return whether all requested semantic fields are present."""
        return _contains_fields(self.fields, _resolve_fields(fields))

    def dtype_for(self, field: TrajectoryField | str) -> pl.DataType | None:
        """Return the canonical dtype for a semantic field if present."""
        trajectory_field = _as_trajectory_field(field)
        if not _contains_fields(self.fields, trajectory_field):
            return None
        return _FIELD_DTYPES[trajectory_field]

    def column_for(self, field: TrajectoryField | str) -> str | None:
        """Return the canonical physical column name for a semantic field."""
        trajectory_field = _as_trajectory_field(field)
        if not _contains_fields(self.fields, trajectory_field):
            return None
        return trajectory_field.to_str()

    def semantic_fields(self) -> tuple[str, ...]:
        """Return the semantic fields physically present in this schema."""
        return tuple(field.to_str() for field in self.ordered_fields())

    def feature_fields(self) -> tuple[TrajectoryField, ...]:
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


def _as_trajectory_field(field: TrajectoryField | str) -> TrajectoryField:
    return field if isinstance(field, TrajectoryField) else TrajectoryField.from_str(field)


def _resolve_fields(fields: TrajectoryField | Iterable[TrajectoryField | str]) -> TrajectoryField:
    if isinstance(fields, TrajectoryField):
        return fields

    resolved = TrajectoryField(0)
    for field in fields:
        resolved |= _as_trajectory_field(field)
    return resolved


def _contains_fields(fields: TrajectoryField, required: TrajectoryField) -> bool:
    return (fields & required) == required


_BASE_FIELDS: Final[TrajectoryField] = (
    TrajectoryField.FRAME
    | TrajectoryField.ID
    | TrajectoryField.AGENT_CATEGORY
    | TrajectoryField.X
    | TrajectoryField.Y
)

POSITIONS_ONLY: Final[TrajectorySchema] = TrajectorySchema.define(
    "positions_only", fields=_BASE_FIELDS
)
POSITIONS_YAW: Final[TrajectorySchema] = TrajectorySchema.define(
    "positions_yaw", fields=_BASE_FIELDS | TrajectoryField.YAW
)
POSITIONS_VELOCITY: Final[TrajectorySchema] = TrajectorySchema.define(
    "positions_velocity", fields=_BASE_FIELDS | TrajectoryField.VX | TrajectoryField.VY
)
POSITIONS_VELOCITY_YAW: Final[TrajectorySchema] = TrajectorySchema.define(
    "positions_velocity_yaw",
    fields=(_BASE_FIELDS | TrajectoryField.VX | TrajectoryField.VY | TrajectoryField.YAW),
)
POSITIONS_VELOCITY_ACCELERATION: Final[TrajectorySchema] = TrajectorySchema.define(
    "positions_velocity_acceleration",
    fields=_BASE_FIELDS
    | TrajectoryField.VX
    | TrajectoryField.VY
    | TrajectoryField.AX
    | TrajectoryField.AY,
)
CANONICAL: Final[TrajectorySchema] = TrajectorySchema.define(
    "canonical",
    fields=_BASE_FIELDS
    | TrajectoryField.VX
    | TrajectoryField.VY
    | TrajectoryField.AX
    | TrajectoryField.AY
    | TrajectoryField.YAW,
)

TRAJECTORY_SCHEMAS: Final[dict[str, TrajectorySchema]] = {
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


def available_trajectory_schemas() -> tuple[TrajectorySchema, ...]:
    """Return all registered built-in trajectory schemas in stable display order."""
    return tuple(TRAJECTORY_SCHEMAS.values())


def available_trajectory_schema_names() -> tuple[str, ...]:
    """Return all registered built-in trajectory schema names."""
    return tuple(TRAJECTORY_SCHEMAS)


class TrajectorySchemaDefinition(TypedDict):
    """Structured payload used to define a custom trajectory schema."""

    name: str
    fields: TrajectoryField | Sequence[TrajectoryField | str]


def get_trajectory_schema(
    schema: TrajectorySchema | str | TrajectorySchemaDefinition,
) -> TrajectorySchema:
    """Resolve a schema instance or registered schema name to a trajectory schema."""
    if isinstance(schema, TrajectorySchema):
        return schema

    if isinstance(schema, dict):
        return TrajectorySchema.define(schema["name"], fields=schema["fields"])

    if schema in TRAJECTORY_SCHEMAS:
        return TRAJECTORY_SCHEMAS[schema]

    msg = f"Unknown trajectory schema '{schema}'."
    raise TrajectorySchemaError(msg)
