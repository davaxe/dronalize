# Contributing

<div class="section-intro" markdown="1">
This is a light first-pass contributor page. It focuses on getting a local environment running and validating changes against the current Python, lint, and docs toolchain.
</div>

## Local setup

<div class="command-block" markdown="1">
```bash
uv sync --group dev --group tools
```
</div>

## Validation commands

<div class="command-block" markdown="1">
```bash
uv run pytest
uv run ruff check .
uv run zensical build --strict
```
</div>

## Contribution posture

- Keep changes small and aligned to one behavioral area when possible.
- Update or add tests when runtime behavior changes.
- Update documentation when the CLI, config shape, or public Python surface shifts.

## Repository context

The project is packaged from `src/dronalize`, uses `uv` for reproducible environments, and keeps docs in the Zensical-managed `docs` tree.
