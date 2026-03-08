from pathlib import Path
from typing import Protocol

from dronalize.core.datatypes.scene import Scene
from dronalize.core.datatypes.split import DatasetSplit


class SceneWriter(Protocol):
    """Protocol for writing processed scenes to disk."""

    def write(self, processed: Scene, split: DatasetSplit | None = None) -> bool:
        """Write a single processed scene.

        Parameters
        ----------
        processed : Scene
            The processed scene to write.
        split : DatasetSplit or None, optional
            Optional information of what dataset split the scene belongs to.

        Returns
        -------
        bool
            Returns `True` if anything was actually written (e.g. the scene
            passed validation and was not

        """
        ...

    def finalize(self) -> None:
        """Finalize the writing process.

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
