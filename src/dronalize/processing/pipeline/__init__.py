"""Core pipeline abstractions used by the processing runtime.

The pipeline package exposes the small set of public types needed to talk about
pipeline shape:

- `Pipeline` for immutable transform composition
- `Transform` for one-to-one lazy-frame transforms
- `FlatMapTransform` for transforms that emit multiple frames per input frame

Most supporting helpers, presets, and functional building blocks stay in
submodules so this root surface remains compact.
"""

from collections.abc import Callable, Iterable

import polars as pl

from dronalize.processing.pipeline.pipeline import Pipeline

Transform = Callable[[pl.LazyFrame], pl.LazyFrame]
"""Signature for a one-to-one LazyFrame transformation used in a pipeline step."""

FlatMapTransform = Callable[[pl.LazyFrame], Iterable[pl.LazyFrame]]
"""Signature for a one-to-many LazyFrame transformation used in a flat-map step."""

__all__ = ["FlatMapTransform", "Pipeline", "Transform"]
