# Installation

<div class="section-intro" markdown="1">
The base package is intentionally small. Start there, then add extras for the CLI, MDS storage,
Torch or PyG adapters, or dataset-specific loaders only when you need them.
</div>

## Requirements

- Python `>=3.10`
- `pip` or `uv`

## Common installs

```bash
pip install dronalize
pip install "dronalize[cli]"
pip install "dronalize[cli,mds]"
pip install "dronalize[torch]"
pip install "dronalize[pyg]"
```

Combine extras as needed in one install command.

## What the base package includes

The base install already gives you:

- the dataset registry and runtime planning API
- the processing pipeline and built-in scene/schema types
- the `pickle` storage backend
- the framework-neutral `PickleReader`

The base install does not include the practical CLI dependencies, the MDS backend, or the optional
Torch and PyG adapters.

## Optional extras

| Extra | Purpose |
| --- | --- |
| `cli` | Enables the Typer and Rich command-line interface. |
| `mds` | Enables the `mds` writer backend and `MDSReader`. |
| `torch` | Enables Torch-based dataset adapters such as `TorchSceneDataset`. |
| `pyg` | Enables PyTorch Geometric adapters such as `HeteroSceneDataset`. |
| `plot` | Installs Altair-based plotting helpers. |
| `lyft`, `waymo`, `ad4che` | Add dataset-specific optional dependencies. |
| `all_datasets` | Convenience extra for all dataset-specific extras. |

## Readers and adapters

| Surface | Extra |
| --- | --- |
| `dronalize.io.readers.PickleReader` | none |
| `dronalize.io.readers.MDSReader` | `mds` |
| `dronalize.io.adapters.TorchSceneDataset` | `torch` |
| `dronalize.io.adapters.HeteroSceneDataset` | `pyg` |

## Dataset extras

Most built-in datasets work with the base install. The dataset registry only exposes entries whose
optional dependencies are currently available, so `dronalize available` reflects the environment
you actually installed.

Use a dataset extra only when a dataset needs one:

| Dataset | Extra |
| --- | --- |
| `waymo` | `waymo` |
| `lyft` | `lyft` |
| `ad4che` | `ad4che` |

## Working with `uv`

```bash
uv sync --group dev --group tools
uv run dronalize available
```

The repository tracks a `uv.lock`, so `uv` is the most direct way to reproduce the local
development and documentation environment.

## Verify the install

```bash
python -c "import dronalize"
```

If you installed the CLI extra, also verify:

```bash
dronalize available
dronalize inspect a43
```

If you installed the MDS extra, a quick import check is:

```bash
python -c "from dronalize.io.readers import MDSReader"
```
