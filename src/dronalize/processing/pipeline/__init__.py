"""Core pipeline abstractions used by the processing runtime.

The pipeline package exposes the small set of public types needed to talk about
pipeline shape:

- `Pipeline` for immutable transform composition
- `Transform` for one-to-one lazy-frame transforms
- `FlatMapTransform` for transforms that emit multiple frames per input frame
- `build_trajectory_pipeline` for the runtime trajectory-processing pipeline

Most supporting helpers, presets, and functional building blocks stay in
submodules so this root surface remains compact.
"""

from dronalize.processing.pipeline.pipeline import FlatMapTransform, Pipeline, Transform
from dronalize.processing.pipeline.trajectory import build_trajectory_pipeline

__all__ = ["FlatMapTransform", "Pipeline", "Transform", "build_trajectory_pipeline"]
