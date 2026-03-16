from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag, auto
from typing import TYPE_CHECKING, Final

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable


class SceneField(IntFlag):
    """Canonical semantic fields supported by scene schemas."""

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
            raise ValueError(msg) from exc

    def fields(self) -> Iterable[SceneField]:
        """Yield individual SceneField members present in this combination."""
        return _ordered_fields(self)

    def to_str(self) -> str:
        """Return the canonical physical column name for this SceneField."""
        if self not in _FIELD_DTYPES:
            msg = f"SceneField value {self.value} does not correspond to a valid field."
            raise ValueError(msg)
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

_REQUIRED_FIELDS: Final[SceneField] = SceneField.FRAME | SceneField.ID | SceneField.X | SceneField.Y


@dataclass(slots=True, frozen=True, init=False)
class SceneSchema:
    """Logical scene schema defined as a set of canonical semantic fields."""

    name: str
    version: int
    fields: SceneField

    def __init__(
        self,
        name: str,
        *,
        fields: SceneField | Iterable[SceneField | str],
        version: int = 1,
    ) -> None:
        resolved_fields = _resolve_fields(fields)
        missing_required = _REQUIRED_FIELDS & ~resolved_fields
        if missing_required:
            missing: str = ", ".join(field.to_str() for field in missing_required.fields())
            msg = f"Scene schemas must include the base fields: {missing}."
            raise ValueError(msg)

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "fields", resolved_fields)

    @classmethod
    def define(
        cls,
        name: str,
        *,
        fields: SceneField | Iterable[SceneField | str],
        version: int = 1,
    ) -> SceneSchema:
        """Construct a schema from semantic field identifiers."""
        return cls(name=name, version=version, fields=fields)

    @property
    def physical(self) -> pl.Schema:
        """Return the canonical physical dataframe schema for this field set."""
        return pl.Schema({field.to_str(): _FIELD_DTYPES[field] for field in self.ordered_fields()})

    def ordered_fields(self) -> tuple[SceneField, ...]:
        """Return semantic fields in canonical dataframe order."""
        return _ordered_fields(self.fields)

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

    def field_items(self) -> tuple[tuple[str, str, pl.DataType], ...]:
        """Return semantic field, physical column, and dtype tuples."""
        return tuple(
            (field.to_str(), field.to_str(), _FIELD_DTYPES[field])
            for field in self.ordered_fields()
        )


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


def _ordered_fields(fields: SceneField) -> tuple[SceneField, ...]:
    return tuple(field for field in SceneField if _contains_fields(fields, field))


_BASE_FIELDS: Final[SceneField] = (
    SceneField.FRAME | SceneField.ID | SceneField.X | SceneField.Y | SceneField.AGENT_CATEGORY
)

POSITIONS_ONLY_V1: Final[SceneSchema] = SceneSchema.define("positions_only", fields=_BASE_FIELDS)
POSITIONS_YAW_V1: Final[SceneSchema] = SceneSchema.define(
    "positions_yaw",
    fields=_BASE_FIELDS | SceneField.YAW,
)
POSITIONS_VELOCITY_V1: Final[SceneSchema] = SceneSchema.define(
    "positions_velocity",
    fields=_BASE_FIELDS | SceneField.VX | SceneField.VY,
)
POSITIONS_VELOCITY_YAW_V1: Final[SceneSchema] = SceneSchema.define(
    "positions_velocity_yaw",
    fields=_BASE_FIELDS | SceneField.VX | SceneField.VY | SceneField.YAW,
)
POSITIONS_VELOCITY_ACCELERATION_V1: Final[SceneSchema] = SceneSchema.define(
    "positions_velocity_acceleration",
    fields=_BASE_FIELDS | SceneField.VX | SceneField.VY | SceneField.AX | SceneField.AY,
)
CANONICAL_V1: Final[SceneSchema] = SceneSchema.define(
    "canonical",
    fields=_BASE_FIELDS
    | SceneField.VX
    | SceneField.VY
    | SceneField.AX
    | SceneField.AY
    | SceneField.YAW,
)
POSITIONS_VELOCITY_ACCELERATION_YAW_V1 = CANONICAL_V1
