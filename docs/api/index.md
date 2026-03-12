# API Reference

This section provides auto-generated reference documentation for all public modules in the `dronalize` package, powered by [mkdocstrings](https://mkdocstrings.github.io/).

!!! tip "Docstring Convention"
    Dronalize uses the **NumPy** docstring style throughout the codebase.

---

## Package Structure

| Module | Description |
|--------|-------------|
| [`core`](core.md) | Domain models, scene/category types, map abstractions, data loading, and exceptions |
| [`config`](config.md) | Configuration loading, dataset parameters, filtering, and map settings |
| [`pipeline`](pipeline.md) | Preprocessing pipeline, transforms, and factory utilities |
| [`datasets`](datasets.md) | Dataset-specific preprocessing implementations and the dataset registry |
| [`converters`](converters.md) | Format converters (NumPy, PyTorch) for preprocessed data |
| [`execution`](execution.md) | Parallel and sequential execution runners and data writers |
| [`plot`](plot.md) | Visualization utilities for trajectories, map graphs, and overlays |

---

## Usage

API reference pages are generated directly from the source code docstrings using `mkdocstrings`. Each page documents the public API of a module, including classes, functions, and their parameters.

```python
import dronalize
```

Browse the modules above or use the search bar to find specific classes and functions.
