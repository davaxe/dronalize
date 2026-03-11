from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dronalize.config.map import (
    AutoExtraction,
    MapExtraction,
    NoneExtraction,
    RadialExtraction,
    RectangularExtraction,
    SquareExtraction,
)
from dronalize.core.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    import numpy.typing as npt

    from dronalize.core.map_graph import MapGraph


def extract_fn(
    extraction: MapExtraction,
) -> Callable[[Scene, MapGraph], MapGraph]:
    """Create an extraction function based on the scene and extraction configuration.

    Parameters
    ----------
    extraction : MapExtraction
        The extraction mode configuration.

    Returns
    -------
    Callable[[MapGraph], MapGraph]
        A function that takes a `MapGraph` and returns an extracted `MapGraph`.
    """
    if isinstance(extraction, NoneExtraction):
        return lambda _s, g: g

    def _fn(s: Scene, g: MapGraph) -> MapGraph:
        return extract_based_on_scene(g, s, extraction)

    return _fn


def extract_based_on_scene(
    map_graph: MapGraph,
    scene: Scene,
    extraction: MapExtraction,
) -> MapGraph:
    """Extract a subgraph based on the scene and extraction configuration.

    Parameters
    ----------
    scene : Scene
        The scene containing relevant positions for auto extraction.
    extraction : MapExtraction
        The extraction mode configuration.

    Returns
    -------
    MapGraph
        The extracted subgraph.
    """
    center_x = scene.inner.select("x").mean().item()
    center_y = scene.inner.select("y").mean().item()
    center = (center_x, center_y)
    return extract(
        map_graph,
        center=center,
        extraction=extraction,
        relevant_positions=scene,
    )


def extract(
    graph: MapGraph,
    center: tuple[float, float] | npt.NDArray[np.floating[Any]] | None,
    extraction: MapExtraction,
    *,
    relevant_positions: npt.NDArray[np.floating[Any]] | Scene | None = None,
) -> MapGraph:
    """Extract a subgraph from the given graph based on the extraction mode.

    Parameters
    ----------
    graph : MapGraph
        The input graph to extract from.
    center : tuple[float, float] | npt.NDArray[np.floating] | None
        The center point of the extraction region, either as typle or array. If
        `None`, the graph's center is used.
    extraction : MapExtraction
        The extraction mode configuration.
    relevant_positions : npt.NDArray[np.floating] | Scene, optional
        Relevant positions to consider for `AutoExtraction` mode, either as a
        NumPy array of shape (N, 2) or a `Scene` containing `x` and `y` columns.

    Returns
    -------
    MapGraph
        The extracted subgraph.
    """
    if isinstance(relevant_positions, Scene):
        relevant_positions = relevant_positions.inner.select("x", "y").to_numpy()

    match extraction:
        case RectangularExtraction(width=width, height=height):
            return graph.extract_bbox(center, width, height)
        case RadialExtraction(radius=radius):
            return graph.extract_radius(center, radius)
        case SquareExtraction(size=size):
            return graph.extract_bbox(center, size, size)
        case AutoExtraction(padding_factor=padding_factor):
            if relevant_positions is None:
                msg = "relevant_positions must be provided for AutoExtraction"
                raise ValueError(msg)
            return graph.extract_relevant(relevant_positions, padding_factor)
        case NoneExtraction():
            return graph
