"""Loader implementation for the uniD dataset."""

from __future__ import annotations

from dronalize.datasets.shared.levelx_loader import LevelXDataLoader


class UniDLoader(LevelXDataLoader):
    """Trajectory data loader for the uniD dataset.

    The uniD (urban intersections Drone) dataset was recorded at urban
    intersections in Germany using drone footage. It contains naturalistic
    trajectories of various traffic participants, including cars, trucks, buses,
    motorcycles, bicycles, and pedestrians navigating through signalised and
    unsignalised intersections.

    This loader inherits all trajectory processing logic from
    `XLevelDataLoader`, as the uniD dataset follows the same CSV format used
    across the X-level dataset family.

    """


if __name__ == "__main__":
    from dronalize.datasets.shared._debug import debug_descriptor, resolve_dataset_root_from_env
    from dronalize.datasets.unid import DATASET_SPEC

    root = resolve_dataset_root_from_env("unid")
    _ = debug_descriptor(DATASET_SPEC, root)
