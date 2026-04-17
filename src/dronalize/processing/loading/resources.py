"""Run-scoped resources injected into dataset loaders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.processing.maps.resolver import MapKey


@dataclass(slots=True, frozen=True)
class DatasetResources:
    """Shared resources prepared once for a processing run.

    The main current use case is shared-memory map lookup tables, but the
    container stays intentionally generic so datasets can grow into additional
    run-scoped resources without reintroducing loader-global state.
    """

    shared_maps: dict[MapKey, str] | str | None = None

    @classmethod
    def empty(cls) -> DatasetResources:
        """Create an empty DatasetResources instance."""
        return cls()
