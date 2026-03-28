from __future__ import annotations

_CLI_INSTALL_HINT = "Install dronalize[cli] to use the dronalize command line interface."
_CLI_DEPENDENCIES = ("typer", "rich")


def _is_missing_cli_dependency(exc: ModuleNotFoundError) -> bool:
    """Check if the missing module is a required CLI dependency."""
    name = exc.name or ""
    return any(name == dep or name.startswith(f"{dep}.") for dep in _CLI_DEPENDENCIES)


def main() -> None:
    """Run the optional CLI if its dependencies are installed."""
    try:
        from dronalize.runtime.cli.app import main as cli_main  # noqa: PLC0415
    except ModuleNotFoundError as exc:
        if _is_missing_cli_dependency(exc):
            raise ModuleNotFoundError(_CLI_INSTALL_HINT) from exc
        raise

    cli_main()


if __name__ == "__main__":
    main()
