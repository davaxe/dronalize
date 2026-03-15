from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from typing_extensions import Self

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from dronalize.categories import DatasetSplit
    from dronalize.scene import Scene


@runtime_checkable
class SceneWriter(Protocol):
    """Protocol for writing processed scenes to disk."""

    def write(
        self,
        scene: Scene,
        split: DatasetSplit | None = None,
    ) -> int:
        """Write a single processed scene.

        Parameters
        ----------
        scene : Scene
            The processed scene to write.
        splits : Iterator[DatasetSplit] or None, optional
            An optional iterator of dataset splits that the scene belongs to.
            An iterator is primarily used to allow for multiple splits per write
            call.
        strict : bool, optional
            Whether to raise an error if the `splits` iterator is longer than
            the total written samples. Defaults to `False`.

        Returns
        -------
        int
            The number of samples written for the given scene.

        """
        ...

    def finish_local(self) -> None:
        """Finalize the writing process for the local process.

        This method can be used to perform any necessary cleanup or finalization
        steps after all scenes have been written. For example, closing file
        handles, writing summary files, or performing any necessary
        post-processing.

        """
        ...

    def set_output_dir(self, output_dir: Path) -> None:
        """Set the output directory for the writer.

        This method can be used to specify the directory where the writer should
        save its output files. This is particularly useful in parallel processing
        contexts, where each process may need to write to a separate directory
        to avoid conflicts.

        Parameters
        ----------
        output_dir : Path
            The path to the output directory where the writer should save its files.

        """
        ...

    def get_output_dir(self) -> Path:
        """Get the current output directory for the writer.

        Returns
        -------
        Path
            The path to the current output directory where the writer is saving its files.

        """
        ...

    def finish_final(self) -> None:
        """Perform any final cleanup after all writing is complete.

        This will be called once in parallel context after all processes have
        completed their writing and finalization steps. For non-parallel context
        this will be called immediately after `finish_local`.

        """
        ...

    @classmethod
    def as_factory(cls, *args: Any, **kwargs: Any) -> Callable[[int | None], Self]:  # noqa: ANN401
        """Create a factory function for this writer class.

        The returned factory should accept a worker identifier and return an
        instance of the writer class ready to be used independently by that
        worker. Most commonly, the identifier is used to shard output paths or
        filenames to avoid write conflicts.

        Parameters
        ----------
        *args : Any (should be specified more concretely in subclasses)
            Positional arguments to be passed to the writer constructor.
        **kwargs : Any (should be specified more concretely in subclasses)
            Keyword arguments to be passed to the writer constructor.

        Returns
        -------
        Callable[[int | None], Self]
            A factory function that takes a worker identifier and returns an
            instance of the writer class.

        """
        return functools.partial(cls, *args, **kwargs)
