import sys
from pathlib import Path

from dronalize.config.project import ProjectConfig
from dronalize.core.errors import ConfigurationError

if sys.version_info >= (3, 11):
    import tomllib  # pyright: ignore[reportUnreachable]
else:
    import tomli as tomllib


def parse_config(path: Path) -> ProjectConfig:
    """Parse configuration from a TOML file.

    !!! note "Completeness of the returned config"
        By design, this loader returns a validated but potentially incomplete configuration for
        specific datasets.

        See the
        [`ProjectConfig.resolve_dataset_config`][dronalize.config.ProjectConfig.resolve_dataset_config]
        method for applying dataset-specific overrides returned by this loader
        on top of a fully resolved `DatasetConfig` to get a complete
        configuration for a specific dataset.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Returns
    -------
    ProjectConfig
        The parsed configuration, validated and ready for dataset-specific resolution.
    """
    with path.open("rb") as handle:
        try:
            data = tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            msg = f"Invalid TOML in config file '{path}': {exc}"
            raise ConfigurationError(msg) from exc
    return ProjectConfig.model_validate(data)
