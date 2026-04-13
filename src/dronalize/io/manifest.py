"""Persisted metadata models shared across storage backends."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


FORMAT_VERSION: int = 1
MANIFEST_FILENAME: str = "manifest.json"


@dataclass(slots=True, frozen=True)
class DatasetManifest:
    """Format-agnostic metadata stored alongside exported datasets."""

    source_trajectory_schema: str
    trajectory_schema: str
    derived_features: tuple[str, ...]
    feature_columns: tuple[str, ...]
    history_frames: int
    future_frames: int
    precision: str
    recenter_positions: bool
    has_map: bool
    sample_time: float
    original_sample_time: float
    format_version: int = FORMAT_VERSION

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the manifest."""
        return asdict(self)

    @classmethod
    def from_json_dict(cls, payload: dict[str, Any]) -> DatasetManifest:
        """Create a manifest from previously serialized JSON data."""
        return cls(
            format_version=int(payload["format_version"]),
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
    payload = json.loads(manifest_path(root).read_text(encoding="utf-8"))
    return DatasetManifest.from_json_dict(payload)
