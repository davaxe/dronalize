# Data formats

<div class="section-intro" markdown="1">
`dronalize` stores processed scenes using a backend-neutral scene model and a backend-specific
on-disk layout. In practice, you choose a trajectory schema and precision once, then pick the
storage backend that best fits your workflow.
</div>

<figure markdown="block" class="figure">
  ![Design architecture](../assets/io-dark.svg#only-dark){ width="100%" }
  ![Design architecture](../assets/io-light.svg#only-light){ width="100%" }
  <figcaption>
    High-level view of the export and reading model.
  </figcaption>
</figure>

## What is persisted

Every processing run produces:

- one root `manifest.json` that describes how scenes were produced and can be
  loaded as a [`DatasetManifest`][dronalize.io.DatasetManifest]
- one or more split directories that contain scene data in the selected backend format

Split directory names follow the split strategy:

- `train`, `val`, `test` when a split strategy emits those partitions
- `unsplit` when no split strategy is active

The manifest is backend-independent metadata. Scene files are backend-specific.

## Core concepts

Two choices define the output dataset:

1. **Trajectory schema**
   Determines which per-timestep features are written (for example position-only vs. velocity + yaw).
2. **Storage backend**
   Determines how scene records are serialized on disk (`pickle`, `mds`, or `null`).

This separation lets you keep one feature definition while switching storage format as needed.

## Documentation map

- [Outputs and schemas](schema.md)
- [Manifest](manifest.md)
- [Reading data](reading.md)
- [Storage backends](backends/index.md)
