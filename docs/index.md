---
hide:
  - navigation
  - toc
  - path
---
<div class="landing" markdown>

<div class="hero">
  <img alt="Dronalize logo" src="assets/dronalize-header.png" class="hero__image">
  <p class="hero__eyebrow">Trajectory preprocessing toolkit</p>

  <div class="hero__badges">
    <a class="hero__badge" href="https://pypi.org/project/dronalize/">
      <img alt="PyPI" src="https://img.shields.io/pypi/v/dronalize">
    </a>
    <a class="hero__badge" href="https://pypi.org/project/dronalize/">
      <img alt="Python versions" src="https://img.shields.io/pypi/pyversions/dronalize">
    </a>
    <a class="hero__badge" href="https://www.apache.org/licenses/LICENSE-2.0">
      <img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg">
    </a>
    <a class="hero__badge" href="https://arxiv.org/abs/2405.00604">
      <img alt="ArXiv link" src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=8A2BE2&logo=arxiv">
    </a>
  </div>
</div>

`dronalize` is a toolbox designed to streamline the development process for researchers working with trajectory datasets in behavior prediction problems. Originally developed for drone-captured bird’s-eye-view datasets, it has since evolved to support a wide range of popular benchmarks in motion forecasting.

The package can be installed via `pip` or `uv`:
```sh
pip install dronalize
```

or using `uv`:

```sh
uv pip install dronalize
```

## Navigation

<div class="grid cards" markdown>

-   [:material-rocket-launch:{ .lg .middle } __Start here__](start/index.md)

    ---

    Install `dronalize`, run the CLI once, and choose whether your workflow starts from the terminal or Python.


-   [:material-source-branch:{ .lg .middle } __Concepts__](concepts/index.md)

    ---

    Understand how dataset specs, configuration layers, screening, splits, maps, and outputs fit together.


-   [:material-database:{ .lg .middle } __Data formats__](formats/index.md)

    ---

    See how processed scenes are stored, what the manifest contains, and which backend or reader model fits your use case.


-   [:material-map-search:{ .lg .middle } __Dataset reference__](reference/datasets/index.md)

    ---

    Browse supported datasets, expected on-disk structure, and source links before preparing raw data locally.


-   [:material-tune:{ .lg .middle } __Configuration reference__](reference/configuration/index.md)

    ---

    Look up the TOML config surface, section layout, and field behavior when you need exact option details.


-   [:material-api:{ .lg .middle } __API reference__](reference/api/index.md)

    ---

    Jump into the generated Python API docs when you already know the area you want to inspect in code.

</div>

</div>
