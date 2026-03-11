# Quickstart

<!-- TODO: Port and expand from _docs content -->

!!! note "Work in Progress"
    This page is a placeholder. Content will be ported from the existing documentation.

## Overview

This page will walk you through a minimal end-to-end workflow with Dronalize: from downloading a dataset to running a training loop.

## Steps

1. **Install Dronalize** — See the [Installation](installation.md) guide.
2. **Download a dataset** — Pick one of the [supported datasets](../user-guide/datasets/index.md).
3. **Preprocess the data** — Run the preprocessing pipeline.
4. **Train a model** — Launch the example training script.

## Preprocessing

```bash
# Example: preprocess the rounD dataset
python -m preprocessing.preprocess_urban --config rounD --path ../datasets --output-dir ./data --use-threads 1
```

## Training

```bash
# Example: train the prototype model
python train.py --add-name Test --dry-run 0 --use-cuda 1 --num-workers 4
```

## Next Steps

- Explore the [User Guide](../user-guide/index.md) for in-depth documentation on each component.
- Check out the [API Reference](../api/index.md) for detailed module documentation.