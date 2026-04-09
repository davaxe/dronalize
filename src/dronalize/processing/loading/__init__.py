"""Advanced loader API for dataset integrations.

## Import guide

```python
from dronalize.processing.loading import BaseSceneLoader, LoaderOptions
from dronalize.processing.loading import LoaderConfig, WindowConfig
```

This package contains the extension hooks used by built-in datasets and custom
dataset integrations.

Use it when you are:

- subclassing [`BaseSceneLoader`][dronalize.processing.loading.BaseSceneLoader]
- defining dataset-specific loader options
- advertising split capabilities for a loader
- working with normalized source or prepared-scene transport types
- wiring advanced split strategies into a dataset integration

For most day-to-day processing code, [`dronalize.processing`][] is the better
entry point. This subpackage is intentionally more detailed because it
documents the loader contract itself.

## Related modules

- [`dronalize.datasets`][] for the dataset registry and
  [`DatasetSpec`][dronalize.datasets.DatasetSpec]
- [`dronalize.processing`][] for the higher-level processing entry point
"""

from dronalize.processing.loading.base import (
    BaseSceneLoader,
    LoaderOptions,
    LoaderSplitCapabilities,
    NoLoaderOptions,
)
from dronalize.processing.loading.config import LaneChangeSamplingConfig, LoaderConfig, WindowConfig
from dronalize.processing.loading.loader import (
    BlockSplitSupport,
    LoadedSourceData,
    MapBinding,
    PreparedSceneData,
    Source,
)
from dronalize.processing.loading.splits import (
    NativeSplitStrategy,
    NoSplitStrategy,
    SceneSplitStrategy,
    ShuffledTimeBlockStrategy,
    SourceSplitStrategy,
    SplitConfig,
    SplitStrategy,
    SplitStrategyName,
    SplitWeights,
    TimeBlockStrategy,
)

__all__ = [
    "BaseSceneLoader",
    "BlockSplitSupport",
    "LaneChangeSamplingConfig",
    "LoadedSourceData",
    "LoaderConfig",
    "LoaderOptions",
    "LoaderSplitCapabilities",
    "MapBinding",
    "NativeSplitStrategy",
    "NoLoaderOptions",
    "NoSplitStrategy",
    "PreparedSceneData",
    "SceneSplitStrategy",
    "ShuffledTimeBlockStrategy",
    "Source",
    "SourceSplitStrategy",
    "SplitConfig",
    "SplitStrategy",
    "SplitStrategyName",
    "SplitWeights",
    "TimeBlockStrategy",
    "WindowConfig",
]
