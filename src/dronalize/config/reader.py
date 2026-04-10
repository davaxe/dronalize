from pathlib import Path

from dronalize.config.project import ProjectConfig

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_project_config(path: Path) -> ProjectConfig:
    """Load and parse a project configuration from a TOML file.

    Parameters
    ----------
    path : Path
        Path to the TOML configuration file.

    Returns
    -------
    ProjectConfig
        The parsed project configuration, fully validated and ready for use.
    """
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return ProjectConfig.model_validate(data)
