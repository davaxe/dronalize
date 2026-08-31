---
hide:
  - navigation
  - toc
  - path
---
<div class="landing" markdown>

<div class="hero">
  <img alt="Prejectory logo" src="assets/prejectory-header.png" class="hero__image">
  <p class="hero__eyebrow">Trajectory preprocessing toolkit</p>

  <div class="hero__badges">
    <a class="hero__badge" href="https://pypi.org/project/prejectory/">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/prejectory">
    </a>
    <a class="hero__badge" href="https://pypi.org/project/prejectory/">
      <img alt="Python versions" src="https://img.shields.io/pypi/pyversions/prejectory">
    </a>
    <a class="hero__badge" href="https://www.apache.org/licenses/LICENSE-2.0">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg">
    </a>
    <a class="hero__badge" href="https://arxiv.org/abs/2405.00604">
      <img alt="ArXiv link" src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=8A2BE2&logo=arxiv">
    </a>
  </div>
</div>

`prejectory` is a toolbox designed to streamline the development process for researchers working with trajectory datasets in behavior prediction problems. Originally developed for drone-captured bird’s-eye-view datasets, it has since evolved to support a wide range of popular benchmarks in motion forecasting.

The package can be installed via `pip` or `uv`:
```sh
pip install prejectory
```

or using `uv`:

```sh
uv pip install prejectory
```

## Navigation

<div class="grid cards" markdown>

-   [:material-rocket-launch:{ .lg .middle } __Install__](start/installation/)

    ---

    Install `prejectory` and the optional extras needed for your datasets, readers, and training stack.


-   [:material-terminal:{ .lg .middle } __First CLI run__](start/first-run-cli/)

    ---

    Inspect a dataset, preview resolved configuration, and run `process --plan` before writing outputs.


-   [:material-language-python:{ .lg .middle } __Python usage__](start/python-entry/)

    ---

    Use the public Python API to inspect datasets, resolve config, plan runs, and execute processing.


-   [:material-map-search:{ .lg .middle } __Dataset reference__](reference/datasets/)

    ---

    Browse supported datasets, expected on-disk structure, and source links before preparing raw data locally.


-   [:material-tune:{ .lg .middle } __Configuration reference__](reference/configuration/)

    ---

    Look up the TOML config surface, section layout, and field behavior when you need exact option details.


-   [:material-api:{ .lg .middle } __API reference__](reference/api/)

    ---

    Jump into the generated Python API docs when you already know the area you want to inspect in code.

</div>

</div>
