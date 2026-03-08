from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Protocol

from dronalize.core.datatypes.scene import Scene


class TrajectoryFormat(StrEnum):
    DATAFRAME = "dataframe"
    """Raw DataFrame format, where each scene is written as a single DataFrame."""
    NUMPY = "numpy"
    """Nmupy format, where each scene is represented by a collection of arrays."""


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


class BaseSceneWriter(ABC, SceneWriter):
    def __init__(self, *, parallel: bool = False) -> None:
        """Initialize the base scene writer.

        Parameters
        ----------
        parallel : bool, optional
            Whether this writer will be used in a parallel processing context.
            This can be used to determine whether the writer needs to handle
            potential write conflicts (e.g., by writing to separate files or
            directories for each process). Default is `False`.

        """
        self._parallel: bool = parallel
        if parallel and not self.support_parallel():
            msg = f"{self.__class__.__name__} does not support parallel writing,"
            " but was initialized with `parallel=True`"
            raise ValueError(msg)

    @abstractmethod
    def write(self, processed: Scene) -> None: ...

    @abstractmethod
    def finalize(self) -> None: ...

    @abstractmethod
    def support_parallel(self) -> bool: ...
