import os
from pathlib import Path

import pytest


@pytest.fixture
def raw_data_root() -> Path:
    """Path to the root directory containing raw dataset files."""
    root = os.environ.get("TRAJ_DATA")
    return Path(root) if root is not None else Path()


@pytest.fixture
def artifact_dir() -> Path:
    """Path to the directory where test artifacts like debug plots will be saved."""
    return Path("test_artifacts")
