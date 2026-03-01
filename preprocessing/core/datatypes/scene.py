from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, Self, TypeVar

import polars as pl

if TYPE_CHECKING:
    from preprocessing.core.datatypes.map_context import MapContext


T_ID = TypeVar("T_ID", bound=(Hashable))


@dataclass(slots=True, frozen=True)
class Scene(Generic[T_ID]):
    """Scene data class wrapping a DataFrame and its identifier.

    The dataframe is expected to at least contain all columns defined in
    `Scene._base_schema()`: `frame`, `id`, `x`, `y`, `vx`, `vy`,
    `ax`, `ay`, `yaw`, and `agent_category`.
    """

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    identifier: T_ID
    """Identifier for the scene (e.g., file name, index, scene name/token)."""
    scene_number: int
    """Unique scene number assigned during processing."""
    input_len: int
    """Number of observed frames."""
    output_len: int
    """Number of predicted frames."""
    map_context: MapContext
    """Map context for the scene, which can be implicit, explicit, loaded, or no map."""

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
