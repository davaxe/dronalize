from __future__ import annotations

from dronalize.core.categories import DatasetSplit


class DronalizeError(Exception):
    """Base class for all exceptions raised by Dronalize."""


class ConfigurationError(ValueError, DronalizeError):
    """Raised when user-provided configuration is invalid or incomplete."""


class LoaderConfigError(ConfigurationError, DronalizeError):
    """Raised when there is an issue with a loader configuration."""


class MissingOptionalDependencyError(DronalizeError):
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


class DatasetRegistryError(ValueError, DronalizeError):
    """Raised when dataset registry metadata is invalid or inconsistent."""


class DatasetNotFoundError(DronalizeError):
    """Raised when a dataset name cannot be resolved from the registry."""

    def __init__(self, dataset_name: str, available_datasets: list[str]) -> None:
        known = ", ".join(available_datasets) or "none"
        super().__init__(f"Unknown dataset '{dataset_name}'. Available datasets: {known}.")
        self.dataset_name: str = dataset_name
        self.available_datasets: tuple[str, ...] = tuple(available_datasets)


class SplitError(ValueError, DronalizeError):
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
        def _display(value: DatasetSplit | str) -> str:
            return value.value if isinstance(value, DatasetSplit) else str(value)

        if isinstance(split, list):
            split_display = ", ".join(_display(item) for item in split)
        else:
            split_display = _display(split)
        super().__init__(f"{loader_name} does not support split '{split_display}'.")
        self.loader_name: str = loader_name
        self.split: list[str] = (
            [_display(item) for item in split] if isinstance(split, list) else [_display(split)]
        )


class SplitStrategyNotSupportedError(SplitError):
    """Raised when a loader does not implement the requested custom split strategy."""

    def __init__(
        self,
        loader_name: str,
        strategy_name: str,
        supported_strategies: tuple[str, ...],
    ) -> None:
        supported_display = ", ".join(supported_strategies) if supported_strategies else "none"
        msg = (
            f"{loader_name} does not support split strategy '{strategy_name}'. "
            f"Supported strategies: {supported_display}."
        )
        super().__init__(msg)
        self.loader_name: str = loader_name
        self.strategy_name: str = strategy_name
        self.supported_strategies: tuple[str, ...] = supported_strategies


class UnsupportedOutputFormatError(ValueError, DronalizeError):
    """Raised when an unknown writer/output format is requested."""

    def __init__(self, output_format: str, supported_formats: tuple[str, ...]) -> None:
        supported = ", ".join(supported_formats)
        super().__init__(
            f"Unsupported output format '{output_format}'. Supported formats: {supported}.",
        )
        self.output_format: str = output_format
        self.supported_formats: tuple[str, ...] = supported_formats
