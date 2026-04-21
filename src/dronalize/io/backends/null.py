"""No-op writer backend.

The ``null`` backend executes the full runtime, counts accepted scenes, and
intentionally skips all persisted scene output. It is useful for tests, dry
runs, configuration validation, and measuring pipeline cost without storage
I/O.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, final

from typing_extensions import override

from dronalize.io.base import DatasetWriter

if TYPE_CHECKING:
    from collections.abc import Callable

    from dronalize.core.scene import Scene


@final
class NullWriter(DatasetWriter):
    """No-op writer used by tests and dry-run execution paths."""

    def __init__(self, identifier: str | int | None = None) -> None:
        self._identifier: str = "UNNAMED" if identifier is None else str(identifier)

    @classmethod
    @override
    def as_factory(cls, *, log: bool = False) -> Callable[[int | None], NullWriter]:
        """Create a factory that shares split counters across workers."""
        return functools.partial(cls)

    @override
    def write(self, scene: Scene) -> None:
        return

    @override
    def finish_local(self) -> None:
        return

    @override
    def finish_final(self) -> None:
        """Optionally print the aggregated split counts for the full run."""
        return
