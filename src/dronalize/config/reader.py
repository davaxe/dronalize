from pathlib import Path

from dronalize.config.file import ProcessingConfig
from dronalize.core.errors import ConfigurationError

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def parse_config(path: Path) -> ProcessingConfig:
    """Parse configuration from a TOML file.

    !!! note "Completeness of the returned config"
        By design, this loader returns a validated but potentially incomplete configuration for
        specific datasets.

        See the
        [`ProcessingConfig.resolve`][dronalize.config.project.ProcessingConfig.resolve]
        method for applying dataset-specific overrides returned by this loader
        on top of a fully resolved `DatasetConfig` to get a complete
        configuration for a specific dataset. Returns

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Returns
    -------
    ProcessingConfig
        The parsed configuration, fully validated and ready for use.
    """
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Invalid TOML in config file '{path}': {exc}"
            raise ConfigurationError(msg) from exc
    return ProcessingConfig.model_validate(data)
