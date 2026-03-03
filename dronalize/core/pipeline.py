"""Composable pipeline for LazyFrame transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Iterable

    import polars as pl

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Transform(Protocol):
    """A single `LazyFrame -> LazyFrame` transformation step."""

    def __call__(self, df: pl.LazyFrame, /) -> pl.LazyFrame:
        """Apply the transform to a LazyFrame."""
        ...


@runtime_checkable
class FlatMapTransform(Protocol):
    """A step that may produce *multiple* LazyFrames from one input.

    Typical example: sliding-window sampling that splits one recording
    into many overlapping scenes.
    """

    def __call__(self, df: pl.LazyFrame, /) -> Iterable[pl.LazyFrame]:
        """Apply the fan-out to a LazyFrame, yielding multiple outputs."""
        ...


# ---------------------------------------------------------------------------
# Internal entry wrappers
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _MapEntry:
    """Wraps a 1:1 Transform."""

    fn: Transform

    def apply(self, inputs: Iterable[pl.LazyFrame]) -> Iterable[pl.LazyFrame]:
        """Apply the transform for each input."""
        for df in inputs:
            yield self.fn(df)


@dataclass(slots=True, frozen=True)
class _FlatMapEntry:
    """Wraps a 1:N FlatMapTransform."""

    fn: FlatMapTransform

    def apply(self, inputs: Iterable[pl.LazyFrame]) -> Iterable[pl.LazyFrame]:
        """Apply the fan-out for each input, yielding all outputs."""
        for df in inputs:
            yield from self.fn(df)


_PipelineEntry = _MapEntry | _FlatMapEntry

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Immutable, composable chain of LazyFrame transformations.

    Build pipelines with :meth:`then` (1:1) and :meth:`then_flat_map`
    (1:N).  Execute with :meth:`run`.

    Pipelines are **immutable** - every builder method returns a *new*
    `Pipeline` instance, so the original can be safely shared or
    extended independently.
    """

    _steps: tuple[_PipelineEntry, ...] = field(default_factory=tuple)

    # -- builder API --------------------------------------------------------

    def then(self, transform: Transform, *, when: bool = True) -> Pipeline:
        """Append a 1:1 transform and return a *new* pipeline.

        Parameters
        ----------
        transform : Callable[[LazyFrame], LazyFrame]
            A callable that maps one LazyFrame to one LazyFrame.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.

        Returns
        -------
        Pipeline
            New pipeline with the transform appended.

        """
        return Pipeline((*self._steps, _MapEntry(transform))) if when else self

    def then_flat_map(self, flat_map: FlatMapTransform, *, when: bool = True) -> Pipeline:
        """Append a 1:N flat-map step and return a *new* pipeline.

        Any subsequent :meth:`then` transforms will be applied to
        **each** LazyFrame produced by this step independently.

        Parameters
        ----------
        flat_map : Callable[[LazyFrame], Iterable[LazyFrame]]
            A callable that maps one LazyFrame to an iterable of
            LazyFrames.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.

        Returns
        -------
        Pipeline
            New pipeline with the fan-out step appended.

        """
        return Pipeline((*self._steps, _FlatMapEntry(flat_map))) if when else self

    def compose(self, other: Pipeline) -> Pipeline:
        """Concatenate *other* pipeline's steps after this one.

        Parameters
        ----------
        other : Pipeline
            Pipeline whose steps are appended.

        Returns
        -------
        Pipeline
            New pipeline combining both step sequences.

        """
        return Pipeline((*self._steps, *other._steps))

    # -- execution ----------------------------------------------------------

    def execute(self, df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        """Execute every step in order, yielding the resulting LazyFrame(s).

        Parameters
        ----------
        df : pl.LazyFrame
            Input data.

        Yields
        ------
        pl.LazyFrame
            One or more transformed LazyFrames.

        """
        current: Iterable[pl.LazyFrame] = (df,)
        for entry in self._steps:
            current = entry.apply(current)
        yield from current

    def execute_single(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Execute the pipeline expecting exactly one output.

        This is a convenience wrapper around :meth:`run` for pipelines
        that contain no fan-out steps.

        Parameters
        ----------
        df : pl.LazyFrame
            Input data.

        Returns
        -------
        pl.LazyFrame
            The single resulting LazyFrame.

        Raises
        ------
        ValueError
            If the pipeline produces zero or more than one output.

        """
        results = list(self.execute(df))
        if len(results) != 1:
            msg = (
                f"Pipeline.run_single() expected exactly 1 output, "
                f"got {len(results)}.  Use run() for fan-out pipelines."
            )
            raise ValueError(msg)
        return results[0]

    # -- introspection ------------------------------------------------------

    def __len__(self) -> int:
        """Return the number of steps in the pipeline."""
        return len(self._steps)

    def __bool__(self) -> bool:
        """Return True when the pipeline contains at least one step."""
        return len(self._steps) > 0

    def __repr__(self) -> str:
        """Return a human-readable representation of the pipeline."""
        parts: list[str] = []
        for entry in self._steps:
            if isinstance(entry, _MapEntry):
                parts.append(f"  .then({_fn_name(entry.fn)})")
            else:
                parts.append(f"  .then_flat_map({_fn_name(entry.fn)})")
        body = "\n".join(parts) if parts else "  (empty)"
        return f"Pipeline(\n{body}\n)"


def _fn_name(fn: object) -> str:
    """Best-effort human-readable name for a callable."""
    name = getattr(fn, "__name__", None) or getattr(fn, "__qualname__", None)
    if name:
        return name
    cls = type(fn)
    if cls.__name__ != "function":
        return cls.__name__
    return repr(fn)
