import pytest

from dronalize.core.scene import Scene
from tests.support import DataFrameBuilder, DataFramePresets, make_scene
from tests.support import scene_df_presets as _scene_df_presets


@pytest.fixture
def scene() -> Scene:
    return make_scene()


@pytest.fixture
def scene_df_builder() -> DataFrameBuilder:
    return DataFrameBuilder()


@pytest.fixture
def scene_df_presets() -> DataFramePresets:
    return _scene_df_presets()
