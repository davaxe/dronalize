"""Composable pipeline for LazyFrame transformations."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, cast, overload

import polars as pl
from typing_extensions import override

if TYPE_CHECKING:
    from collections.abc import Iterator

Transform = Callable[[pl.LazyFrame], pl.LazyFrame]
"""A 1:1 transformation from one LazyFrame to another."""

FlatMapTransform = Callable[[pl.LazyFrame], Iterable[pl.LazyFrame]]
"""A 1:N transformation that maps one LazyFrame to an iterable of LazyFrames."""


@dataclass(frozen=True, slots=True)
class Pipeline:
    """Immutable, composable chain of LazyFrame transformations.

    Build pipelines with `then` (1:1) and `then_flat_map`
    (1:N).  Execute with `run`.

    Pipelines are **immutable** - every builder method returns a *new*
    `Pipeline` instance, so the original can be safely shared or
    extended independently.
    """

    _steps: tuple[_PipelineEntry, ...] = field(default_factory=tuple)

    # -- builder API --------------------------------------------------------

    def then(
        self,
        transform: Transform,
        *,
        when: bool = True,
        otherwise: Transform | None = None,
        name: str | None = None,
    ) -> Pipeline:
        """Append a 1:1 transform and return a *new* pipeline.

        Parameters
        ----------
        transform : Transform
            A callable that maps one LazyFrame to one LazyFrame.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.
        otherwise : Transform, optional
            If provided and `when` is `False`, this transform will be used
            instead of skipping the step.
        name : str | None, optional
            An optional name for this step, used in the pipeline's repr.

        Returns
        -------
        Pipeline
            New pipeline with the transform appended.

        """
        if not when:
            if otherwise is None:
                return self
            transform = otherwise
        return Pipeline((*self._steps, _MapEntry(transform, name)))

    def then_flat_map(
        self,
        transform: FlatMapTransform,
        *,
        when: bool = True,
        otherwise: FlatMapTransform | None = None,
        name: str | None = None,
    ) -> Pipeline:
        """Append a 1:N flat-map step and return a *new* pipeline.

        Any subsequent `then` transforms will be applied to
        **each** LazyFrame produced by this step independently.

        Parameters
        ----------
        transform : FlatMapTransform
            A callable that maps one LazyFrame to an iterable of
            LazyFrames.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.
        otherwise : FlatMapTransform, optional
            If provided and `when` is `False`, this flat-map will be used
            instead of skipping the step.
        name : str | None, optional
            An optional name for this step, used in the pipeline's repr.

        Returns
        -------
        Pipeline
            New pipeline with the flat-map step appended.

        """
        if not when:
            if otherwise is None:
                return self
            transform = otherwise
        return Pipeline((*self._steps, _FlatMapEntry(transform, name)))

    def compose(
        self, other: Pipeline, *, when: bool = True, otherwise: Pipeline | None = None
    ) -> Pipeline:
        """Concatenate *other* pipeline's steps after this one.

        Parameters
        ----------
        other : Pipeline
            Pipeline whose steps are appended.
        when : bool, optional
            If False, skip this step and return the unchanged pipeline.
        otherwise : Pipeline, optional
            If provided and `when` is `False`, this pipeline will be returned
            instead of the unchanged pipeline.

        Returns
        -------
        Pipeline
            New pipeline combining both step sequences.

        """
        if not when:
            if otherwise is None:
                return self
            return otherwise
        return Pipeline((*self._steps, *other._steps))

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
        self, df: pl.LazyFrame | pl.DataFrame, *, collect: Literal[True], filter_empty: bool = True
    ) -> Iterator[pl.DataFrame]: ...

    def execute(
        self, df: pl.LazyFrame | pl.DataFrame, *, collect: bool = False, filter_empty: bool = True
    ) -> Iterator[pl.LazyFrame | pl.DataFrame]:
        """
        Execute every step in order, yielding the resulting LazyFrame(s).

        Parameters
        ----------
        df : pl.LazyFrame or pl.DataFrame
            Input data. DataFrames will be converted to LazyFrames before
            processing.
        collect : bool, optional
            If `True, collect each resulting LazyFrame to a DataFrame before
            yielding.
        filter_empty : bool, optional
            If `True`, skip any resulting DataFrames that are empty. Only
            applies when `collect=True`.

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
        self, df: pl.LazyFrame, *, collect: Literal[True], filter_empty: Literal[True]
    ) -> pl.DataFrame | None: ...

    def execute_single(
        self, df: pl.LazyFrame | pl.DataFrame, *, collect: bool = False, filter_empty: bool = True
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

        result = cast("pl.DataFrame", result)  # type checker cannot infer this
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

    def cast(self, **dtypes: pl.DataType) -> Pipeline:
        """Append a cast step to the pipeline.

        This is a common enough pattern that it gets its own convenience method.
        Equivalent to `then(lambda df: df.select(pl.col("*").cast(**dtypes)))`.

        Parameters
        ----------
        **dtypes : pl.datatypes.Dtype
            Mapping of column names to dtypes to cast.

        Returns
        -------
        Pipeline
            New pipeline with the cast step appended.
        """
        keys = tuple(dtypes.keys())
        targets = tuple(dtypes.values())

        def transform(df: pl.LazyFrame) -> pl.LazyFrame:
            return df.with_columns(pl.col(*keys).cast(*targets))

        return self.then(transform, name="cast")

    # -- introspection ------------------------------------------------------

    def inspect(self, f: Callable[[pl.LazyFrame], None]) -> Pipeline:
        """Apply a side-effecting function to each pipeline step.

        This is useful for debugging or logging the steps in a pipeline.

        Parameters
        ----------
        f : Callable[[LazyFrame], None]
            A function that takes a pipeline entry and performs side effects.

        Returns
        -------
        Pipeline
            The (functionally) unchanged pipeline, for chaining.
        """

        def _inspect_transform(df: pl.LazyFrame) -> pl.LazyFrame:
            f(df)
            return df

        return self.then(_inspect_transform, name="inspect")

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

    @override
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

    def __rshift__(self, other: Pipeline) -> Pipeline:
        """Compose two pipelines.

        This operator is a convenient shorthand for `compose`, allowing you to
        write `p1 >> p2` instead of `p1.compose(p2)`.

        """
        return self.compose(other)


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


def _fn_name(fn: object) -> str:
    """Best-effort human-readable name for a callable."""
    name = getattr(fn, "__name__", None) or getattr(fn, "__qualname__", None)
    if name:
        return name
    cls = type(fn)
    if cls.__name__ != "function":
        return cls.__name__
    return repr(fn)
