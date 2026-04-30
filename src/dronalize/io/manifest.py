"""Persisted metadata models shared across storage backends."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from dronalize.core.errors import ManifestCompatibilityError

if TYPE_CHECKING:
    from pathlib import Path


FORMAT_VERSION: int = 1
MANIFEST_FILENAME: str = "manifest.json"
logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class DatasetManifest:
    """Format-agnostic metadata stored alongside exported datasets.

    The manifest records the shape and schema contract of one processed export.
    Reader and adapter code use it to understand feature columns, temporal
    horizons, coordinate handling, map availability, and manifest compatibility.
    """

    source_trajectory_schema: str
    """Trajectory schema emitted by the dataset loader before conversion."""
    trajectory_schema: str
    """Trajectory schema stored in the exported records."""
    derived_features: tuple[str, ...]
    """Output features derived during schema conversion."""
    feature_columns: tuple[str, ...]
    """Per-timestep feature columns stored in record tensors."""
    history_frames: int
    """Number of observation frames per record."""
    future_frames: int
    """Number of prediction frames per record."""
    precision: str
    """Floating-point precision used for exported feature arrays."""
    recenter_positions: bool
    """Whether spatial values were recentered around each scene."""
    has_map: bool
    """Whether records may contain map topology arrays."""
    sample_time: float
    """Output sample interval in seconds after resampling."""
    original_sample_time: float
    """Dataset sample interval in seconds before resampling."""
    format_version: int = FORMAT_VERSION
    """Manifest schema version used for compatibility checks."""

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the manifest."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        """Create a manifest from previously serialized JSON data."""
        format_version = int(payload.get("format_version", 0))
        if format_version != FORMAT_VERSION:
            raise ManifestCompatibilityError(format_version, FORMAT_VERSION)
        return cls(
            format_version=format_version,
            source_trajectory_schema=str(
                payload.get("source_trajectory_schema", payload["trajectory_schema"])
            ),
            trajectory_schema=str(payload["trajectory_schema"]),
            derived_features=tuple(payload.get("derived_features", ())),
            feature_columns=tuple(payload["feature_columns"]),
            history_frames=int(payload["history_frames"]),
            future_frames=int(payload["future_frames"]),
            precision=str(payload["precision"]),
            recenter_positions=bool(payload["recenter_positions"]),
            has_map=bool(payload["has_map"]),
            sample_time=float(payload["sample_time"]),
            original_sample_time=float(payload["original_sample_time"]),
        )


def manifest_path(root: Path) -> Path:
    """Return the manifest path for one storage root.

    Parameters
    ----------
    root: Path
        The root directory of the processed dataset.

    Returns
    -------
    Path
        The path to the manifest file.
    """
    return root / MANIFEST_FILENAME


def write_manifest(root: Path, manifest: DatasetManifest) -> None:
    """Write the storage manifest for one output root."""
    root.mkdir(parents=True, exist_ok=True)
    logger.debug("Writing manifest", extra={"root": str(root), "format_version": FORMAT_VERSION})
    _ = manifest_path(root).write_text(
        json.dumps(manifest.to_json_dict(), indent=2), encoding="utf-8"
    )


def read_manifest(root: Path) -> DatasetManifest:
    """Read and parse the storage manifest for one output root.

    Parameters
    ----------
    root: Path
        The root directory of the processed dataset.

    Returns
    -------
    DatasetManifest
        The parsed dataset manifest.
    """
    logger.debug("Reading manifest", extra={"root": str(root)})
    payload = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    return DatasetManifest.from_json_dict(payload)
