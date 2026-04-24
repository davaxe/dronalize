from __future__ import annotations

from typing import TYPE_CHECKING, Any, overload

import numpy as np
import numpy.typing as npt

from dronalize.config.models import (
    BoundingBoxExtraction,
    CircularExtraction,
    FullMapExtraction,
    MapConfig,
    MapEdgeTypesConfig,
    MapExtraction,
    SceneExtentExtraction,
    TrajectoryBufferExtraction,
)
from dronalize.core.categories import EdgeType, EdgeTypeLike
from dronalize.core.maps import MapGraph
from dronalize.core.scene import Scene

if TYPE_CHECKING:
    from collections.abc import Callable


def extract_fn(extraction: MapExtraction) -> Callable[[Scene, MapGraph], MapGraph]:
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
    if isinstance(extraction, FullMapExtraction):
        return lambda _s, g: g

    def _fn(s: Scene, g: MapGraph) -> MapGraph:
        return extract_based_on_scene(g, s, extraction)

    return _fn


def extract_based_on_scene(
    map_graph: MapGraph, scene: Scene, extraction: MapExtraction
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
    center_x = scene.frame.select("x").mean().item()
    center_y = scene.frame.select("y").mean().item()
    center = (center_x, center_y)
    return extract(map_graph, center=center, extraction=extraction, relevant_positions=scene)


def apply_map_config(map_graph: MapGraph, config: MapConfig) -> MapGraph:
    """Apply config-wide map transforms that do not depend on a scene."""
    return apply_edge_type_config(map_graph, config.edge_types)


def extract_configured_map(map_graph: MapGraph, scene: Scene, config: MapConfig) -> MapGraph:
    """Apply map config and then extract the scene-local subgraph."""
    return extract_based_on_scene(apply_map_config(map_graph, config), scene, config.extraction)


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
        Relevant positions to consider for `SceneExtentExtraction` mode, either as a
        NumPy array of shape (N, 2) or a `Scene` containing `x` and `y` columns.

    Returns
    -------
    MapGraph
        The extracted subgraph.
    """
    if isinstance(relevant_positions, Scene):
        relevant_positions = relevant_positions.frame.select("x", "y").to_numpy()

    match extraction:
        case BoundingBoxExtraction(width=width, height=height):
            return graph.extract_bbox(center, width, height)
        case CircularExtraction(radius=radius):
            return graph.extract_radius(center, radius)
        case TrajectoryBufferExtraction(radius=radius):
            if relevant_positions is None:
                msg = "relevant_positions must be provided for TrajectoryBufferExtraction"
                raise ValueError(msg)
            return graph.extract_trajectory_buffer(relevant_positions, radius)
        case SceneExtentExtraction(padding=padding, shape=shape):
            if relevant_positions is None:
                msg = "relevant_positions must be provided for SceneExtentExtraction"
                raise ValueError(msg)
            return graph.extract_relevant(
                relevant_positions, padding, use_bbox=shape == "bounding_box"
            )
        case FullMapExtraction():
            return graph


def apply_edge_type_config(map_graph: MapGraph, edge_types: MapEdgeTypesConfig | None) -> MapGraph:
    """Apply edge-type remapping and filtering to a graph."""
    if edge_types is None or map_graph.num_edges == 0:
        return map_graph

    include, exclude, remap = _normalize_edge_type_rules(edge_types)
    remapped_edge_types = np.array(map_graph.edge_types, copy=True)
    changed = False
    for source, target in remap.items():
        source_value = int(source)
        target_value = int(target)
        matches = remapped_edge_types == source_value
        if matches.any():
            remapped_edge_types[matches] = target_value
            changed = True

    edge_mask = np.ones(map_graph.num_edges, dtype=bool)
    if include is not None:
        include_values = np.array([int(edge_type) for edge_type in include], dtype=np.int32)
        edge_mask &= np.isin(remapped_edge_types, include_values)
    if exclude:
        exclude_values = np.array([int(edge_type) for edge_type in exclude], dtype=np.int32)
        edge_mask &= ~np.isin(remapped_edge_types, exclude_values)

    if not edge_mask.all():
        filtered_graph = map_graph.filter_edges(edge_mask)
        filtered_edge_types = remapped_edge_types[edge_mask]
        return MapGraph(
            node_positions=filtered_graph.node_positions,
            edge_indices=filtered_graph.edge_indices,
            node_types=filtered_graph.node_types,
            edge_types=filtered_edge_types,
        )

    if not changed:
        return map_graph

    return MapGraph(
        node_positions=map_graph.node_positions,
        edge_indices=map_graph.edge_indices,
        node_types=map_graph.node_types,
        edge_types=remapped_edge_types,
    )


def _normalize_edge_type_rules(
    edge_types: MapEdgeTypesConfig,
) -> tuple[frozenset[EdgeType] | None, frozenset[EdgeType], dict[EdgeType, EdgeType]]:
    include = (
        None
        if edge_types.include is None
        else frozenset(EdgeType.from_value(edge_type) for edge_type in edge_types.include)
    )
    exclude = frozenset(EdgeType.from_value(edge_type) for edge_type in edge_types.exclude)
    remap = {
        EdgeType.from_value(source): EdgeType.from_value(target)
        for source, target in edge_types.remap.items()
    }
    return include, exclude, remap


FloatArray = npt.NDArray[np.floating[Any]]

_K0 = 0.9996

_E = 0.00669438
_E2 = _E * _E
_E3 = _E2 * _E
_E_P2 = _E / (1 - _E)

_SQRT_E = np.sqrt(1 - _E)
__E = (1 - _SQRT_E) / (1 + _SQRT_E)
__E2 = __E * __E
__E3 = __E2 * __E
__E4 = __E3 * __E
__E5 = __E4 * __E

_M1 = 1 - _E / 4 - 3 * _E2 / 64 - 5 * _E3 / 256
_M2 = 3 * _E / 8 + 3 * _E2 / 32 + 45 * _E3 / 1024
_M3 = 15 * _E2 / 256 + 45 * _E3 / 1024
_M4 = 35 * _E3 / 3072

_R = 6378137
_ZONE_LETTERS = "CDEFGHJKLMNPQRSTUVWXX"


@overload
def from_latlon(
    latitude: float,
    longitude: float,
    *,
    force_zone_number: int | None = None,
    force_zone_letter: str | None = None,
    force_northern: bool | None = None,
) -> tuple[float, float, int, str | None]: ...


@overload
def from_latlon(
    latitude: FloatArray,
    longitude: FloatArray,
    *,
    force_zone_number: int | None = None,
    force_zone_letter: str | None = None,
    force_northern: bool | None = None,
) -> tuple[FloatArray, FloatArray, int, str | None]: ...


def from_latlon(
    latitude: float | FloatArray,
    longitude: float | FloatArray,
    *,
    force_zone_number: int | None = None,
    force_zone_letter: str | None = None,
    force_northern: bool | None = None,
) -> tuple[float | FloatArray, float | FloatArray, int, str | None]:
    """Convert latitude and longitude coordinates to UTM coordinates.

    This function is ported from `utm.from_latlon` with modification for better
    type hints. Reason for porting instead of depending on the library is to
    avoid an extra dependency just for this function.

    """
    latitude_array, longitude_array = np.broadcast_arrays(
        np.asarray(latitude, dtype=np.float64), np.asarray(longitude, dtype=np.float64)
    )

    if not _in_bounds(latitude_array, -80, 84):
        msg = "latitude out of range (must be between 80 deg S and 84 deg N)"
        raise ValueError(msg)
    if not _in_bounds(longitude_array, -180, 180):
        msg = "longitude out of range (must be between 180 deg W and 180 deg E)"
        raise ValueError(msg)
    if force_zone_letter and force_northern is not None:
        msg = "set either force_zone_letter or force_northern, but not both"
        raise ValueError(msg)
    if force_zone_number is not None:
        _check_valid_zone(force_zone_number, force_zone_letter)

    lat_rad = np.radians(latitude_array)
    lat_sin = np.sin(lat_rad)
    lat_cos = np.cos(lat_rad)

    lat_tan = lat_sin / lat_cos
    lat_tan2 = lat_tan * lat_tan
    lat_tan4 = lat_tan2 * lat_tan2

    zone_number = (
        _latlon_to_zone_number(latitude_array, longitude_array)
        if force_zone_number is None
        else force_zone_number
    )

    zone_letter = (
        _latitude_to_zone_letter(latitude_array)
        if force_zone_letter is None and force_northern is None
        else force_zone_letter.upper()
        if force_zone_letter is not None
        else None
    )
    northern = (
        (zone_letter is not None and zone_letter >= "N")
        if force_northern is None
        else force_northern
    )

    lon_rad = np.radians(longitude_array)
    central_lon = _zone_number_to_central_longitude(zone_number)
    central_lon_rad = np.radians(central_lon)

    n = _R / np.sqrt(1 - _E * lat_sin**2)
    c = _E_P2 * lat_cos**2

    a = lat_cos * _mod_angle(lon_rad - central_lon_rad)
    a2 = a * a
    a3 = a2 * a
    a4 = a3 * a
    a5 = a4 * a
    a6 = a5 * a

    m = _R * (
        _M1 * lat_rad
        - _M2 * np.sin(2 * lat_rad)
        + _M3 * np.sin(4 * lat_rad)
        - _M4 * np.sin(6 * lat_rad)
    )

    easting = (
        _K0
        * n
        * (
            a
            + a3 / 6 * (1 - lat_tan2 + c)
            + a5 / 120 * (5 - 18 * lat_tan2 + lat_tan4 + 72 * c - 58 * _E_P2)
        )
        + 500000
    )

    northing = _K0 * (
        m
        + n
        * lat_tan
        * (
            a2 / 2
            + a4 / 24 * (5 - lat_tan2 + 9 * c + 4 * c**2)
            + a6 / 720 * (61 - 58 * lat_tan2 + lat_tan4 + 600 * c - 330 * _E_P2)
        )
    )

    if force_northern is None and force_zone_letter is None and _mixed_signs(latitude_array):
        msg = "latitudes must all have the same sign"
        raise ValueError(msg)
    if not northern:
        northing += 10000000

    return _to_native(easting), _to_native(northing), zone_number, zone_letter


def _to_native(value: FloatArray) -> float | FloatArray:
    if value.ndim == 0:
        return float(value.item())
    return value


def _in_bounds(
    values: FloatArray, lower: float, upper: float, *, upper_strict: bool = False
) -> bool:
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    if upper_strict:
        return lower <= minimum and maximum < upper
    return lower <= minimum and maximum <= upper


def _check_valid_zone(zone_number: int, zone_letter: str | None) -> None:
    _check_valid_zone_number(zone_number)
    if zone_letter is not None:
        _check_valid_zone_letter(zone_letter)


def _check_valid_zone_letter(zone_letter: str) -> None:
    zone_letter = zone_letter.upper()
    if not ("C" <= zone_letter <= "X") or zone_letter in {"I", "O"}:
        msg = "zone letter out of range (must be between C and X)"
        raise ValueError(msg)


def _check_valid_zone_number(zone_number: int) -> None:
    if not 1 <= zone_number <= 60:
        msg = "zone number out of range (must be between 1 and 60)"
        raise ValueError(msg)


def _mixed_signs(values: FloatArray) -> bool:
    return bool(np.min(values) < 0 and np.max(values) >= 0)


def _mod_angle(value: float | FloatArray) -> float | FloatArray:
    """Return angles in radians normalized to the range [-pi, pi)."""
    return (value + np.pi) % (2 * np.pi) - np.pi


def _latitude_to_zone_letter(latitude: FloatArray) -> str | None:
    latitude_value = float(latitude.flat[0])
    if -80 <= latitude_value <= 84:
        return _ZONE_LETTERS[int(latitude_value + 80) >> 3]
    return None


def _latlon_to_zone_number(latitude: FloatArray, longitude: FloatArray) -> int:
    latitude_value = float(latitude.flat[0])
    longitude_value = float(longitude.flat[0])
    longitude_value = (longitude_value % 360 + 540) % 360 - 180

    if 56 <= latitude_value < 64 and 3 <= longitude_value < 12:
        return 32
    if 72 <= latitude_value <= 84 and longitude_value >= 0:
        if longitude_value < 9:
            return 31
        if longitude_value < 21:
            return 33
        if longitude_value < 33:
            return 35
        if longitude_value < 42:
            return 37

    return int((longitude_value + 180) / 6) + 1


def _zone_number_to_central_longitude(zone_number: int) -> int:
    _check_valid_zone_number(zone_number)
    return (zone_number - 1) * 6 - 180 + 3
