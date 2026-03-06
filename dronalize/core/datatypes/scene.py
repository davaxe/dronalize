from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import polars as pl

if TYPE_CHECKING:
    from dronalize.core.datatypes.map_context import MapKey


IdT = TypeVar("IdT", bound=(Hashable))


@dataclass(slots=True, frozen=True)
class Scene(Generic[IdT]):
    """Scene data class wrapping a DataFrame and its identifier.

    The dataframe is expected to at least contain all columns defined in
    `Scene._base_schema()`: `frame`, `id`, `x`, `y`, `vx`, `vy`,
    `ax`, `ay`, `yaw`, and `agent_category`.
    """

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    identifier: IdT
    """Identifier for the scene (e.g., file name, index, scene name/token)."""
    scene_number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""
    map_key: MapKey = None
    """Lightweight map identifier for the scene.

    A `None` value means "no map" or "use the default/only map".  A
    non-`None` string is resolved by a
    `~dronalize.core.datatypes.map_context.MapResolver` (typically
    obtained from the loader that produced this scene) to produce a
    `~dronalize.core.datatypes.map_graph.MapGraph`.
    """

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
            f"Scene(identifier={self.identifier!r}, "
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
