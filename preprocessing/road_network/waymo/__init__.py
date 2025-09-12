"""Waymo road network processing module."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from preprocessing.road_network.edge_type import EdgeType
from preprocessing.road_network.waymo.parser import WaymoScenario

if TYPE_CHECKING:
    from collections.abc import Iterable

FilterStr = Literal[
    "*training.tfrecord*",
    "*training_20s.tfrecord*",
    "*validation.tfrecord*",
    "validation_interactive.tfrecord*",
    "*testing.tfrecord*",
    "*testing_interactive.tfrecord*",
    "*.tfrecord*",
]


def get_waymo_scenarios_from_directory(
    directory: Path | str,
    filter_str: FilterStr = "*training.tfrecord*",
    *,
    skip_empty_maps: bool = False,
) -> Iterable[WaymoScenario]:
    """Get Waymo scenarios from a directory containing TFRecord files.

    Args:
        directory: Path to the directory containing TFRecord files.
        filter_str: A glob pattern to filter the TFRecord files.
        skip_empty_maps: If True, skip scenarios with empty maps.

    Returns:
        Iterable[WaymoScenario]: An iterable of Waymo scenario objects.

    Yields:
        WaymoScenario: A Waymo scenario object containing the scenario ID,
            map, and scenario data.

    """
    if isinstance(directory, str):
        directory = Path(directory)

    for tfrecord_path in directory.glob(filter_str):
        yield from get_waymo_scenarios_from_tfrecord(
            tfrecord_path,
            skip_empty_maps=skip_empty_maps,
        )


def get_waymo_scenarios_from_tfrecord(
    tfrecord_path: Path | str,
    *,
    skip_empty_maps: bool = False,
) -> Iterable[WaymoScenario]:
    """Get Waymo scenarios from a TFRecord file.

    Args:
        tfrecord_path: Path to the TFRecord file containing Waymo scenarios.
        skip_empty_maps: If True, skip scenarios with empty maps.

    Returns:
        Iterable[WaymoScenario]: An iterable of Waymo scenario objects.

    Yields:
        WaymoScenario: A Waymo scenario object containing the scenario ID,
            map, and scenario data.

    """
    if isinstance(tfrecord_path, str):
        tfrecord_path = Path(tfrecord_path)

    for record in _read_tfrecord(tfrecord_path):
        scenario = WaymoScenario.from_data(record)
        if skip_empty_maps and scenario.map.empty_map():
            continue

        yield scenario


def _read_tfrecord(path: Path) -> Iterable[bytes]:
    """Read a TFRecord file and yield its records.

    Args:
        path: Path to the TFRecord file.

    Returns:
        Iterable[bytes]: An iterable of raw data records from the TFRecord file.

    Yields:
        bytes: The raw data of each record in the TFRecord file.

    """
    file = path.open("rb")
    while True:
        len_bytes = file.read(8)
        if not len_bytes:
            break  # End of file

        length = struct.unpack("<Q", len_bytes)[0]
        file.read(4)  # Skip CRC of length
        data = file.read(length)
        file.read(4)  # Skip CRC of data
        yield data

    file.close()


if __name__ == "__main__":
    from pathlib import Path

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    data = Path("data/training.tfrecord-00000-of-01000")
    scenarios = iter(get_waymo_scenarios_from_tfrecord(data))
    scenario_map = next(scenarios).map
    map_graph = scenario_map.build(gt=2.5)
    print(map_graph)

    # Plot the map
    edge_index = map_graph.edge_indices
    segments = []
    colors = []
    for i in range(edge_index.shape[1]):
        edge_type = EdgeType(map_graph.edge_types[i].item())
        src = map_graph.node_positions[edge_index[0, i]]
        dst = map_graph.node_positions[edge_index[1, i]]
        segments.append([src, dst])
        colors.append(edge_type.edge_style()["color"])

    lines = LineCollection(segments=segments, colors=colors, linewidths=1)

    plt.gca().add_collection(lines)
    plt.axis("equal")

    plt.show()
