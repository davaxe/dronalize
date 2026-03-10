from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Self

import polars as pl

from dronalize.core.datatypes.map_resolver import MapResolver  # noqa: TC001
from dronalize.pipeline.ops.convert import (
    NumpySceneDict,
    convert_to_numpy_dict,
    target_candidates,
)

if TYPE_CHECKING:
    from dronalize.core._types import SceneId
    from dronalize.core.datatypes.map_resolver import MapGraph, MapKey


@dataclass(slots=True, frozen=True)
class Scene:
    """Scene data class wrapping a DataFrame and its identifier.

    The dataframe is expected to at least contain all columns defined in
    `Scene._base_schema()`: `frame`, `id`, `x`, `y`, `vx`, `vy`,
    `ax`, `ay`, `yaw`, and `agent_category`.
    """

    inner: pl.DataFrame
    """Inner DataFrame containing the scene data."""
    identifier: SceneId
    """Identifier for the scene (e.g., file name, index, scene name/token)."""
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
        return self.map_resolver(self.map_key)

    def has_map(self) -> bool:
        """Check if this scene has an attached map resolver and key."""
        return self.map_resolver is not None and self.map_key is not None

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

    def to_numpy_dict(
        self,
        *,
        multiple_targets: bool | int = False,
        target_agent: int | None = None,
    ) -> dict[int, NumpySceneDict]:
        """Convert to a numpy representation compatible with Pytorch.

        Parameters
        ----------
        multiple_targets : bool or int, optional
            Whether to return multile "samples" from each scene by changing the
            target angent. If `True` as many as possible samples will be
            returned, if an integer is given, at most that many samples will be
            returned.
        target_agent : int, optional
            If `multiple_targets` is False, this specifies the track ID to use as
            the target agent. If None, the first valid track will be used as the
            target.

        Returns
        -------
        dict[int, NumpySceneDict]
            A dictionary mapping target agent track IDs to their corresponding
            numpy representations of the scene data.

        """
        if not multiple_targets and target_agent is not None:
            return {
                target_agent: convert_to_numpy_dict(
                    self.inner, self.input_len, self.output_len, target_agent
                )
            }

        candidates = target_candidates(self.inner, self.input_len)
        if multiple_targets is False:
            candidates = candidates[:1]
        elif isinstance(multiple_targets, int):
            candidates = candidates[:multiple_targets]

        return {
            target: convert_to_numpy_dict(
                self.inner,
                self.input_len,
                self.output_len,
                target,
            )
            for target in candidates
        }

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
