from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Self

import polars as pl

from dronalize.core.map_resolver import MapResolver  # noqa: TC001

if TYPE_CHECKING:
    from dronalize.core.map_graph import MapGraph
    from dronalize.core.map_resolver import MapKey


@dataclass(slots=True, frozen=True)
class Scene:
    """Scene data class wrapping a DataFrame and its identifier.

    The dataframe is expected to at least contain all columns defined in
    `Scene._base_schema()`: `frame`, `id`, `x`, `y`, `vx`, `vy`,
    `ax`, `ay`, `yaw`, and `agent_category`.
    """

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    scene_number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""
    map_key: MapKey = None
    """Lightweight map identifier for the scene."""
    map_resolver: MapResolver | None = field(default=None, compare=False, repr=False)
    """Resolver attached by the loader that produced this scene."""

    def resolve_map(self) -> MapGraph | None:
        """Resolve this scene's `map_key` into a `MapGraph`.

        Delegates to the `map_resolver` attached by the loader. Returns `None`
        when no resolver is present or when the resolver has no map for this key
        (e.g. `include_map=False`
        on Waymo).

        Returns
        -------
        MapGraph or None
            The map graph for this scene, or `None` if unavailable.

        """
        if self.map_resolver is None:
            return None
        return self.map_resolver(self, self.map_key)

    def has_map(self) -> bool:
        """Check if this scene has an attached map resolver and key."""
        return self.map_resolver is not None

    def enforce_schema(self, schema: pl.Schema | None = None) -> Self:
        """Enforce the scene dataframe to follow a specified schema.

        This will select relevant columns and try to cast if needed/possible.
        If it is not possible to enforce the schema, an error will be raised.

        Parameters
        ----------
        schema : pl.Schema, optional
            Schema to follow. If None, the base schema is used.

        Returns
        -------
        Self
            Scene with the enforced schema.

        """
        if schema is None:
            schema = Scene._base_schema()
        return replace(
            self,
            inner=self.inner.select([pl.col(name).cast(dtype) for name, dtype in schema.items()]),
        )

    def __repr__(self) -> str:
        """Return a compact representation with metadata and DataFrame shape."""
        rows, cols = self.inner.shape
        return (
            f"scene_number={self.scene_number}, "
            f"input_len={self.input_len}, "
            f"output_len={self.output_len}, "
            f"map_key={self.map_key!r}, "
            f"inner=DataFrame({rows} rows x {cols} cols))"
        )

    @staticmethod
    def _base_schema() -> pl.Schema:
        return pl.Schema({
            "frame": pl.UInt32(),
            "id": pl.Int32(),
            "x": pl.Float32(),
            "y": pl.Float32(),
            "vx": pl.Float32(),
            "vy": pl.Float32(),
            "ax": pl.Float32(),
            "ay": pl.Float32(),
            "yaw": pl.Float32(),
            "agent_category": pl.Int32(),
        })
