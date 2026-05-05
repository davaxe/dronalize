<div align="center">
  <img alt="Dronalize header" src="docs/assets/dronalize-header.png" width="800", style="max-width: 100%; height: auto;">

______________________________________________________________________

[![CI](https://github.com/davaxe/dronalize/actions/workflows/ci.yml/badge.svg)](https://github.com/davaxe/dronalize/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/dronalize)](https://pypi.org/project/dronalize/)
[![Python](https://img.shields.io/pypi/pyversions/dronalize)](https://pypi.org/project/dronalize/)
[![License](https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg)](LICENSE)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=Paper&color=8A2BE2&logo=arxiv)](https://arxiv.org/abs/2405.00604)

</div>

`dronalize` is a trajectory data processing toolkit for autonomous driving datasets.
It provides a consistent pipeline for preparing, inspecting, and exporting trajectory data, with support for both command-line and Python-driven workflows.

The full guides, reference material, and dataset documentation live on the [documentation site](https://westny.github.io/dronalize/).

## Install

```bash
pip install dronalize
pip install "dronalize[cli]"
pip install "dronalize[cli,mds]"
```

The base package includes the runtime API and the `pickle` backend. Add `cli` for the command-line interface and `mds` for Mosaic Streaming output.

## CLI Quickstart

```bash
dronalize available
dronalize inspect a43
dronalize process a43 \
  --input data/a43/raw \
  --output data/a43/processed \
  --config config.toml \
  --plan
```

Use `--plan` first to resolve the run without writing output. When the summary looks right, rerun the same command without `--plan`.

## Python Quickstart

```python
from pathlib import Path

from dronalize.runtime import ExecutionRequest, execute_request, resolve_request

request = ExecutionRequest(
    dataset="a43",
    input_dir=Path("data/a43/raw"),
    output_dir=Path("data/a43/processed"),
    storage_backend="pickle",
)

plan = resolve_request(request)
print(plan.effective_sample_time)

result = execute_request(request)
print(result.selected_scenes)
```

The public Python surface is intentionally explicit:

- `dronalize.datasets` for dataset lookup and descriptors
- `dronalize.config` for TOML loading and runtime overrides
- `dronalize.runtime` for request resolution and execution
- `dronalize.io` for manifests, readers, and adapters

## Cite

If you use the toolbox in your research, please consider citing the paper:

```
@inproceedings{westny2025dronalize,
  title={Toward Unified Practices in Trajectory Prediction Research on Bird's-Eye-View Datasets},
  author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  year={2025}
}
```
