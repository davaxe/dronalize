# ruff: noqa: TC001, TC003, PLR0402

from __future__ import annotations

import collections.abc as cabc
import functools
import pathlib
from typing import Any, Protocol, runtime_checkable

from typing_extensions import Self

import dronalize.categories as categories
import dronalize.scene as scene_module


@runtime_checkable
class SceneWriter(Protocol):
    """Protocol for writing processed scenes to disk.

    Writer implementations are expected to return `True` when a scene produced
    one or more persisted samples and `False` otherwise.
    """

    def write(
        self,
        scene: scene_module.Scene,
        split: categories.DatasetSplit | None = None,
    ) -> bool:
        """Write a single processed scene.

        Parameters
        ----------
        scene : Scene
            The processed scene to write.
        split : DatasetSplit | None, optional
            Explicit dataset split for the scene. When omitted, writers may use
            `scene.split_assignment` instead.

        Returns
        -------
        bool
            `True` if the scene produced output, otherwise `False`.

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

    def set_output_dir(self, output_dir: pathlib.Path) -> None:
        """Set the output directory for the writer.

        This method can be used to specify the directory where the writer should
        save its output files. This is particularly useful in parallel processing
        contexts, where each process may need to write to a separate directory
        to avoid conflicts.

        Parameters
        ----------
        output_dir : pathlib.Path
            The path to the output directory where the writer should save its files.

        """
        ...

    def get_output_dir(self) -> pathlib.Path:
        """Get the current output directory for the writer.

        Returns
        -------
        pathlib.Path
            The path to the current output directory where the writer is saving
            its files.

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
    def as_factory(cls, *args: Any, **kwargs: Any) -> cabc.Callable[[int | None], Self]:  # noqa: ANN401
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
        collections.abc.Callable[[int | None], Self]
            A factory function that takes a worker identifier and returns an
            instance of the writer class.

        """
        return functools.partial(cls, *args, **kwargs)
