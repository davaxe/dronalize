from __future__ import annotations

from dataclasses import dataclass

from dronalize.core.scene.schema import FieldStr, TrajectoryField, TrajectorySchema


@dataclass(frozen=True, slots=True)
class TrajectoryColumns:
    """Column mapping used by trajectory-processing stages."""

    frame: str = "frame"
    agent_id: str = "id"
    category: str = "agent_category"
    x: str | None = None
    y: str | None = None
    vx: str | None = None
    vy: str | None = None
    ax: str | None = None
    ay: str | None = None
    yaw: str | None = None

    @classmethod
    def from_schema(cls, schema: TrajectorySchema | TrajectoryField) -> TrajectoryColumns:
        """Return a column mapping derived from a trajectory schema."""
        if isinstance(schema, TrajectoryField):
            schema = TrajectorySchema(name="", fields=schema)

        return cls(
            frame=TrajectoryColumns._require(schema, TrajectoryField.FRAME),
            agent_id=TrajectoryColumns._require(schema, TrajectoryField.ID),
            category=TrajectoryColumns._require(schema, TrajectoryField.AGENT_CATEGORY),
            x=schema.column_for(TrajectoryField.X),
            y=schema.column_for(TrajectoryField.Y),
            vx=schema.column_for(TrajectoryField.VX),
            vy=schema.column_for(TrajectoryField.VY),
            ax=schema.column_for(TrajectoryField.AX),
            ay=schema.column_for(TrajectoryField.AY),
            yaw=schema.column_for(TrajectoryField.YAW),
        )

    @staticmethod
    def _require(schema: TrajectorySchema, field: TrajectoryField) -> str:
        col = schema.column_for(field)
        if col is None:
            msg = f"Trajectory schema is missing required field: {field}"
            raise ValueError(msg)
        return col

    def get(self, field: FieldStr | TrajectoryField) -> str | None:
        """Return the column name for a given field, or None if not present."""
        if isinstance(field, TrajectoryField):
            field = field.to_str()
        return getattr(self, field, None)

    def require(self, field: FieldStr | TrajectoryField) -> str:
        """Return the column name for a given required field, or raise if not present."""
        col = self.get(field)
        if col is None:
            msg = f"Trajectory columns are missing required field: {field}"
            raise ValueError(msg)
        return col
