# Dronalize

**Trajectory data processing toolkit for autonomous driving datasets.**

Dronalize is a toolbox designed to streamline the development process for researchers working with trajectory datasets in behavior prediction problems. Originally developed for drone-captured bird's-eye-view datasets, it has since evolved to support a wide range of popular benchmarks in motion forecasting.

The toolbox provides utilities for data preprocessing, visualization, and evaluation, along with a model development pipeline tailored for ML-based trajectory prediction.

---

## Key Features

- **Unified preprocessing** — Standardized pipelines for 20+ trajectory datasets
- **Visualization tools** — Plot trajectories, map graphs, and overlays
- **Evaluation metrics** — Comprehensive suite of prediction metrics (minADE, minFDE, Miss Rate, NLL, and more)
- **Modular pipeline** — Configurable transforms, map builders, and data loaders
- **ML-ready** — Built on PyTorch, PyTorch Geometric, and PyTorch Lightning

---

## Quick Links

- **[Getting Started](getting-started/index.md)** — Install Dronalize and run your first preprocessing pipeline.
- **[User Guide](user-guide/index.md)** — Learn about preprocessing, data loading, modeling, and training.
- **[API Reference](api/index.md)** — Auto-generated reference documentation for all public modules.
- **[Community](community/index.md)** — Contribute, cite, or explore related research.

---

## Supported Frameworks

Dronalize builds on top of the following frameworks:

- [PyTorch](https://pytorch.org/) — Deep learning framework
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) — Graph neural network library
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) — Training framework

---

## License

Dronalize is released under the [Apache 2.0 License](community/license.md).
