import os
from pathlib import Path

import pytest

from dronalize.core.scene import Scene
from tests.support import DataFramePresets, make_scene
from tests.support import scene_df_presets as _scene_df_presets


@pytest.fixture
def scene() -> Scene:
    return make_scene()


@pytest.fixture
def scene_df_presets() -> DataFramePresets:
    return _scene_df_presets()


@pytest.fixture
def raw_data_root() -> Path:
    """Path to the root directory containing raw dataset files."""
    root = os.environ.get("TRAJ_DATA")
    return Path(root) if root is not None else Path("data")


@pytest.fixture
def artifact_dir() -> Path:
    """Path to the directory where test artifacts like debug plots will be saved."""
    return Path("test_artifacts")
