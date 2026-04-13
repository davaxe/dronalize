from pathlib import Path

from dronalize.config.project import ProcessingConfig

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_project_config(path: Path) -> ProcessingConfig:
    """Load and parse a project configuration from a TOML file.

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
        The parsed project configuration, fully validated and ready for use.
    """
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return ProcessingConfig.model_validate(data)
