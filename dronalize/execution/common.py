from enum import IntEnum


class ProgressBar(IntEnum):
    """Enum to specify which progress bars to show."""

    NONE = 0
    """No progress bars."""
    SOURCES = 1
    """Show progress bar for sources (# processed scenes are shown in postfix)."""
    SCENES = 2
    """Show progress bar for scenes (# processed sources are shown in postfix)."""

    def unit(self) -> str:
        """Return unit depending on the type of progress bar.

        Returns
        -------
        str
            Unit string for the tqdm progress bar.

        """
        if self == ProgressBar.SOURCES:
            return " sources"
        if self == ProgressBar.SCENES:
            return " scenes"
        return ""
