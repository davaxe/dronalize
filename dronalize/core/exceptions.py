from __future__ import annotations


class DronalizeError(Exception):
    """Base exception class for all dronalize errors."""


class LoaderConfigError(DronalizeError, ValueError):
    """Raised when there is an issue with the loader configuration."""
