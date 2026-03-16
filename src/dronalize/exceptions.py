from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dronalize.categories import DatasetSplit


class ConfigurationError(ValueError):
    """Raised when user-provided configuration is invalid or incomplete."""


class LoaderConfigError(ConfigurationError):
    """Raised when there is an issue with a loader configuration."""


class MissingOptionalDependencyError(ImportError):
    """Raised when an optional dependency required by a feature is unavailable."""

    def __init__(
        self,
        message: str,
        *,
        dependencies: tuple[str, ...] = (),
        install_target: str | None = None,
    ) -> None:
        super().__init__(message)
        self.dependencies: tuple[str, ...] = dependencies
        self.install_target: str | None = install_target


class DatasetRegistryError(ValueError):
    """Raised when dataset registry metadata is invalid or inconsistent."""


class DatasetNotFoundError(Exception):
    """Raised when a dataset name cannot be resolved from the registry."""

    def __init__(self, dataset_name: str, available_datasets: list[str]) -> None:
        known = ", ".join(available_datasets) or "none"
        super().__init__(f"Unknown dataset '{dataset_name}'. Available datasets: {known}.")
        self.dataset_name: str = dataset_name
        self.available_datasets: tuple[str, ...] = tuple(available_datasets)


class SplitError(ValueError):
    """Base class for dataset split validation errors."""


class SplitConflictError(SplitError):
    """Raised when mutually exclusive split options are combined."""


class SplitNotSupportedError(SplitError):
    """Raised when a dataset does not support the requested split.

    This error is raised when one or more specific splits are requested from a
    dataset or loader that does not provide them.

    Parameters
    ----------
    loader_name : str
        The name of the loader or dataset that does not support the split.
    split : DatasetSplit or str or list[DatasetSplit | str]
        The requested split or splits.

    """

    def __init__(
        self,
        loader_name: str,
        split: DatasetSplit | str | list[DatasetSplit] | list[str],
    ) -> None:
        if isinstance(split, list):
            split_display = ", ".join(str(item) for item in split)
        else:
            split_display = str(split)
        super().__init__(f"{loader_name} does not support split '{split_display}'.")
        self.loader_name: str = loader_name
        self.split: list[str] = (
            [str(item) for item in split] if isinstance(split, list) else [str(split)]
        )


class UnsupportedOutputFormatError(ValueError):
    """Raised when an unknown writer/output format is requested."""

    def __init__(self, output_format: str, supported_formats: tuple[str, ...]) -> None:
        supported = ", ".join(supported_formats)
        super().__init__(
            f"Unsupported output format '{output_format}'. Supported formats: {supported}."
        )
        self.output_format: str = output_format
        self.supported_formats: tuple[str, ...] = supported_formats
