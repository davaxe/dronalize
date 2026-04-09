# Installation

<div class="section-intro" markdown="1">
Dronalize now ships its default processing stack in the core package. Start with the base install, then add only the optional extras you actually need for CLI convenience, plotting, PyG adapters, or dataset-specific loaders.
</div>

## Requirements

- Python `>=3.10`
- A local environment with `pip` or `uv`
- Dataset-specific extras only for the datasets you plan to preprocess

## Install profiles

```bash
pip install dronalize
pip install "dronalize[cli]"
pip install "dronalize[plot]"
```

The base install already includes:

- the runtime planning and processing stack
- the default MDS writer backend
- the framework-neutral MDS reader `dronalize.io.readers.mds.MDSReader`
- the Torch adapter `dronalize.io.adapters.MDSTorchDataset`
- PyTorch, which is required by the MDS backend

The current optional extras are:

| Extra | Purpose |
| --- | --- |
| `cli` | Installs Typer and Rich for the command-line application. |
| `plot` | Installs Altair for map and trajectory plotting helpers. |
| `pyg` | Adds PyTorch Geometric for the MDS PyG dataset adapter. |
| `lyft`, `waymo`, `ad4che` | Add dataset-specific optional dependencies. |
| `all_datasets` | Convenience extra for the dataset-specific extras. |

## Dataset extras

Use the base install unless a dataset explicitly needs an extra:

| Dataset or feature | Extra |
| --- | --- |
| `dronalize.io.readers.mds.MDSReader` | none |
| `dronalize.io.adapters.MDSTorchDataset` | none |
| `a43`, `apolloscape`, `argoverse1`, `argoverse2`, `eth`, `exid`, `highd`, `hotel`, `i80`, `ind`, `interact`, `nuscenes`, `opendd`, `round`, `sind`, `unid`, `univ`, `us101`, `vod`, `zara1`, `zara2` | none |
| `waymo` | `waymo` |
| `lyft` | `lyft` |
| `ad4che` | `ad4che` |
| `dronalize.io.adapters.MDSHeteroDataset` | `pyg` |

Use `MDSReader` when you want framework-neutral `RawSceneRecord` objects. Use
`MDSTorchDataset` when you want an iterable Torch dataset surface over the same scene records.
Add the `pyg` extra only if you want `MDSHeteroDataset`, which converts them into PyTorch
Geometric `HeteroData` objects.

## Working with `uv`

```bash
uv sync --group dev --group tools
uv run dronalize available
```

The repository already tracks a `uv.lock`, so `uv` is the most direct way to reproduce the local docs and development environment.

## Verify the install

```bash
dronalize available
dronalize inspect a43
```

If the CLI is not installed, the Python package can still be imported and used programmatically.
