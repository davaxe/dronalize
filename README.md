<div align="center">
  <img alt="Prejectory header" src="https://raw.githubusercontent.com/davaxe/prejectory/dev/docs/assets/prejectory-header.png" width="800", style="max-width: 100%; height: auto;">

______________________________________________________________________

[![CI](https://github.com/davaxe/prejectory/actions/workflows/ci.yml/badge.svg)](https://github.com/davaxe/prejectory/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/prejectory)](https://pypi.org/project/prejectory/)
[![Python](https://img.shields.io/pypi/pyversions/prejectory)](https://pypi.org/project/prejectory/)
[![License](https://img.shields.io/badge/License-Apache%202.0-2F2F2F.svg)](LICENSE)
[![Arxiv link](https://img.shields.io/static/v1?label=arXiv&message=Paper&color=8A2BE2&logo=arxiv)](https://arxiv.org/abs/2405.00604)

</div>

`prejectory` is a trajectory data processing toolkit for autonomous driving datasets.
It provides a consistent pipeline for preparing, inspecting, and exporting trajectory data, with support for both command-line and Python-driven workflows.

The full guides, reference material, and dataset documentation live on the [documentation site](https://davaxe.github.io/prejectory/).

## Install

```bash
pip install prejectory
pip install "prejectory[cli]"
pip install "prejectory[cli,mds]"
```

The base package includes the runtime API and the `pickle` backend. Add `cli` for the command-line interface and `mds` for Mosaic Streaming output.

## CLI Quickstart

```bash
prejectory available
prejectory inspect a43
prejectory process a43 \
  --input data/a43/raw \
  --output data/a43/processed \
  --config config.toml \
  --plan
```

Use `--plan` first to resolve the run without writing output. When the summary looks right, rerun the same command without `--plan`.

## Python Quickstart

```python
from pathlib import Path

from prejectory.runtime import ExecutionRequest, execute_request, resolve_request

request = ExecutionRequest(
    dataset="a43",
    input_dir=Path("data/a43/raw"),
    output_dir=Path("data/a43/processed"),
    storage_backend="pickle",
)

plan = resolve_request(request)
print(plan.effective_sample_time)

result = execute_request(request)
print(result.written_scenes)
```

The public Python surface is intentionally explicit:

- `prejectory.datasets` for dataset lookup and descriptors
- `prejectory.config` for TOML loading and runtime overrides
- `prejectory.runtime` for request resolution and execution
- `prejectory.io` for manifests, readers, and adapters

## Cite

If you use the toolbox in your research, please consider citing the paper:

```
@inproceedings{westny2025prejectory,
  title={Toward Unified Practices in Trajectory Prediction Research on Bird's-Eye-View Datasets},
  author={Westny, Theodor and Olofsson, Bj{\"o}rn and Frisk, Erik},
  booktitle={IEEE Intelligent Vehicles Symposium (IV)},
  year={2025}
}
```
