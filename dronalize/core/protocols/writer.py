from typing import Protocol

from dronalize.core.datatypes.scene import Scene


class SceneWriter(Protocol):
    """Protocol for writing processed scenes to disk."""

    def write(self, processed: Scene) -> None:
        """Write a single processed scene."""
        ...

    def finalize(self) -> None:
        """Finalize the writing process.

        This method can be used to perform any necessary cleanup or finalization
        steps after all scenes have been written. For example, closing file
        handles, writing summary files, or performing any necessary
        post-processing.

        """
        ...


class ParallellSceneWriter(SceneWriter, Protocol):
    """Protocol for writing processed scenes to disk in parallel."""

    def write(self, processed: Scene) -> None:
        """Write a single processed scene."""
        ...

    def finalize(self) -> None:
        """Finalize the writing process.

        This method can be used to perform any necessary cleanup or finalization
        steps after all scenes have been written. For example, closing file
        handles, writing summary files, or performing any necessary
        post-processing.

        """
        ...
