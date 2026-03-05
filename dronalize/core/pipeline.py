"""Composable pipeline for LazyFrame transformations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, runtime_checkable

from typing_extensions import overload

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    import polars as pl

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class Transform(Protocol):
    """A single LazyFrame -> LazyFrame transformation step."""

    def __call__(self, df: pl.LazyFrame) -> pl.LazyFrame:
        """Apply the transform to a LazyFrame."""
        ...


@runtime_checkable
class FlatMapTransform(Protocol):
    """A step that may produce *multiple* LazyFrames from one input.

    Typical example: sliding-window sampling that splits one recording
    into many overlapping scenes.
    """

    def __call__(self, df: pl.LazyFrame) -> Iterable[pl.LazyFrame]:
        """Apply the flat-map to a LazyFrame, yielding multiple outputs."""
        ...


# ---------------------------------------------------------------------------
# Internal entry wrappers
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class _MapEntry:
    """Wraps a 1:1 Transform."""

    fn: Transform
    name: str | None = None

    def apply(self, inputs: Iterable[pl.LazyFrame]) -> Iterable[pl.LazyFrame]:
        """Apply the transform for each input."""
        for df in inputs:
            yield self.fn(df)


@dataclass(slots=True, frozen=True)
class _FlatMapEntry:
    """Wraps a 1:N FlatMapTransform."""

    fn: FlatMapTransform
    name: str | None = None

    def apply(self, inputs: Iterable[pl.LazyFrame]) -> Iterable[pl.LazyFrame]:
        """Apply the flat-map for each input, yielding all outputs."""
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

    def then(self, transform: Transform, *, when: bool = True, name: str | None = None) -> Pipeline:
        """Append a 1:1 transform and return a *new* pipeline.

        Parameters
        ----------
        transform : Callable[[LazyFrame], LazyFrame]
            A callable that maps one LazyFrame to one LazyFrame.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.
        name : str | None, optional
            An optional name for this step, used in the pipeline's repr.

        Returns
        -------
        Pipeline
            New pipeline with the transform appended.

        """
        return Pipeline((*self._steps, _MapEntry(transform, name))) if when else self

    def then_lazy(
        self,
        transform_factory: Callable[..., Transform],
        *,
        when: bool = True,
        name: str | None = None,
    ) -> Pipeline:
        """Like `then` but the transform is produced by a factory at build time.

        This is useful for deferring expensive setup until it's known to be needed.

        Parameters
        ----------
        transform_factory : Callable[..., Transform]
            A factory that produces a Transform when called.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.
        name : str | None, optional
            An optional name for this step, used in the pipeline's repr.

        Returns
        -------
        Pipeline
            New pipeline with the transform appended if when is True, else the
            unchanged pipeline.

        """
        if not when:
            return self
        transform = transform_factory()
        return self.then(transform, name=name)

    def then_if_present(
        self,
        transform_factory: Callable[[T], Transform],
        arg: T | None,
    ) -> Pipeline:
        """Like `then` but only append the transform if *arg* is not None.

        Parameters
        ----------
        transform_factory : Callable[[T], Transform]
            A factory that produces a Transform when given an argument.
        arg : T or None
            The argument to pass to the factory. If None, this step is skipped.

        Returns
        -------
        Pipeline
            New pipeline with the transform appended if arg is not None, else
            the unchanged pipeline.
        """
        if arg is None:
            return self
        transform = transform_factory(arg)
        return self.then(transform)

    def then_flat_map(
        self,
        flat_map: FlatMapTransform,
        *,
        when: bool = True,
        name: str | None = None,
    ) -> Pipeline:
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
        name : str | None, optional
            An optional name for this step, used in the pipeline's repr.

        Returns
        -------
        Pipeline
            New pipeline with the flat-map step appended.

        """
        return Pipeline((*self._steps, _FlatMapEntry(flat_map, name))) if when else self

    def then_lazy_flat_map(
        self,
        flat_map_factory: Callable[..., FlatMapTransform],
        *,
        when: bool = True,
        name: str | None = None,
    ) -> Pipeline:
        """Like `then_flat_map` but the flat-map is produced by a factory at build time.

        This is useful for deferring expensive setup until it's known to be needed.

        Parameters
        ----------
        flat_map_factory : Callable[..., FlatMapTransform]
            A factory that produces a FlatMapTransform when called.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.
        name : str | None, optional
            An optional name for this step, used in the pipeline's repr.

        Returns
        -------
        Pipeline
            New pipeline with the flat-map step appended if when is True, else the
            unchanged pipeline.

        """
        if not when:
            return self
        flat_map = flat_map_factory()
        return self.then_flat_map(flat_map, name=name)

    def then_if_present_flat_map(
        self,
        flat_map_factory: Callable[[T], FlatMapTransform],
        arg: T | None,
    ) -> Pipeline:
        """Like `then_flat_map` but only append the flat-map if *arg* is not None.

        Parameters
        ----------
        flat_map_factory : Callable[[T], FlatMapTransform]
            A factory that produces a FlatMapTransform when given an argument.
        arg : T or None
            The argument to pass to the factory. If None, this step is skipped.

        Returns
        -------
        Pipeline
            New pipeline with the flat-map step appended if arg is not None, else
            the unchanged pipeline.
        """
        if arg is None:
            return self
        flat_map = flat_map_factory(arg)
        return self.then_flat_map(flat_map)

    def compose(self, other: Pipeline, *, when: bool = True) -> Pipeline:
        """Concatenate *other* pipeline's steps after this one.

        Parameters
        ----------
        other : Pipeline
            Pipeline whose steps are appended.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.

        Returns
        -------
        Pipeline
            New pipeline combining both step sequences.

        """
        return Pipeline((*self._steps, *other._steps)) if when else self

    # -- slicing ------------------------------------------------------------

    def take(self, n: int) -> Pipeline:
        """Take only the first n steps of the pipeline.

        Parameters
        ----------
        n : int
            Number of steps to take. Must be less than the total number of
            steps.

        Returns
        -------
        Pipeline
            New pipeline containing only the first n steps.

        Raises
        ------
        ValueError
            If n is greater than or equal to the total number of steps in the
            pipeline.
        """
        if len(self._steps) <= n:
            msg = f"Pipeline.take({n}) called on pipeline with only {len(self._steps)} steps"
            raise ValueError(msg)

        return Pipeline(self._steps[:n])

    # -- execution ----------------------------------------------------------

    @overload
    def execute(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        collect: Literal[False] = False,
        filter_empty: bool = True,
    ) -> Iterator[pl.LazyFrame]: ...

    @overload
    def execute(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        collect: Literal[True],
        filter_empty: bool = True,
    ) -> Iterator[pl.DataFrame]: ...

    def execute(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        collect: bool = False,
        filter_empty: bool = True,
    ) -> Iterator[pl.LazyFrame | pl.DataFrame]:
        """
        Execute every step in order, yielding the resulting LazyFrame(s).

        Parameters
        ----------
        df : pl.LazyFrame or pl.DataFrame
            Input data. DataFrames will be converted to LazyFrames before
            processing.
        collect : bool, optional
            If True, collect each resulting LazyFrame to a DataFrame before
            yielding.
        filter_empty : bool, optional
            If True, skip any resulting DataFrames that are empty.
            Only applies when `collect=True`.

        Yields
        ------
        pl.LazyFrame | pl.DataFrame
            One or more transformed frames.
        """
        current: Iterable[pl.LazyFrame] = (df.lazy(),)

        for entry in self._steps:
            current = entry.apply(current)

        if not collect:
            yield from current
            return

        for lf in current:
            df_collected = lf.collect()

            if filter_empty and df_collected.height == 0:
                continue

            yield df_collected

    @overload
    def execute_single(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        collect: Literal[False] = False,
        filter_empty: bool = True,
    ) -> pl.LazyFrame: ...

    @overload
    def execute_single(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        collect: Literal[True],
        filter_empty: Literal[False] = False,
    ) -> pl.DataFrame: ...

    @overload
    def execute_single(
        self,
        df: pl.LazyFrame,
        *,
        collect: Literal[True],
        filter_empty: Literal[True],
    ) -> pl.DataFrame | None: ...

    def execute_single(
        self,
        df: pl.LazyFrame | pl.DataFrame,
        *,
        collect: bool = False,
        filter_empty: bool = True,
    ) -> pl.LazyFrame | pl.DataFrame | None:
        """
        Execute the pipeline expecting exactly one output.

        This is a convenience wrapper for pipelines that contain no flat-map steps.

        Parameters
        ----------
        df : pl.LazyFrame or pl.DataFrame
            Input data. DataFrames will be converted to LazyFrames before
            processing.
        collect : bool, optional
            If True, collect the resulting LazyFrame.
        filter_empty : bool, optional
            If True and collect=True, return None if the collected
            DataFrame is empty.

        Returns
        -------
        pl.LazyFrame | pl.DataFrame | None
            The single resulting frame. May return None if
            collect=True and filter_empty=True and the result is empty.

        Raises
        ------
        ValueError
            If the pipeline produces zero or more than one output.
        """
        results = list(self.execute(df, collect=collect, filter_empty=False))

        if len(results) != 1:
            msg = (
                f"Pipeline.execute_single() expected exactly 1 output, "
                f"got {len(results)}. Use execute() for flat-map pipelines."
            )
            raise ValueError(msg)

        result = results[0]

        if not collect:
            return result

        if filter_empty and result.height == 0:
            return None

        return result

    # -- common polars patterns ---------------------------------------------

    def with_columns(self, *exprs: pl.Expr, **named_expr: pl.Expr) -> Pipeline:
        """Append a with_columns step to the pipeline.

        This is a common enough pattern that it gets its own convenience method.
        Equivalent to `then(lambda df: df.with_columns(*exprs, **named_expr))`.

        Parameters
        ----------
        *exprs : pl.Expr
            see `pl.LazyFrame.with_columns`.
        **named_expr : pl.Expr
            see `pl.LazyFrame.with_columns`.

        Returns
        -------
        Pipeline
            New pipeline with the with_columns step appended.
        """

        def transform(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(*exprs, **named_expr)

        return self.then(transform, name="with_columns")

    def select(self, *exprs: pl.Expr, **named_expr: pl.Expr) -> Pipeline:
        """Append a select step to the pipeline.

        This is a common enough pattern that it gets its own convenience method.
        Equivalent to `then(lambda df: df.select(*exprs, **named_expr))`.

        Parameters
        ----------
        *exprs : pl.Expr
            see `pl.LazyFrame.select`.
        **named_expr : pl.Expr
            see `pl.LazyFrame.select`.

        Returns
        -------
        Pipeline
            New pipeline with the select step appended.
        """

        def transform(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.select(*exprs, **named_expr)

        return self.then(transform, name="select")

    # -- introspection ------------------------------------------------------

    def is_empty(self) -> bool:
        """Return True if the pipeline contains no steps."""
        return len(self._steps) == 0

    def has_flat_map(self) -> bool:
        """Return True if the pipeline contains any flat-map steps."""
        return any(isinstance(entry, _FlatMapEntry) for entry in self._steps)

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
                parts.append(f"  .then({entry.name or _fn_name(entry.fn)})")
            else:
                parts.append(f"  .then_flat_map({entry.name or _fn_name(entry.fn)})")
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
