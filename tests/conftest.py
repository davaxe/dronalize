from collections.abc import Iterable

import polars as pl
import pytest

from dronalize.core.scene import Scene
from tests.support import AgentData, DataFrameBuilder, DataFramePresets, make_scene, make_scene_df
from tests.support import scene_df_presets as _scene_df_presets


@pytest.fixture
def scene() -> Scene:
    return make_scene()


@pytest.fixture
def scene_df_builder() -> DataFrameBuilder:
    def _build_scene_df(agent_data: Iterable[AgentData]) -> pl.DataFrame:
        return make_scene_df(*agent_data)

    return _build_scene_df


@pytest.fixture
def scene_df_presets() -> DataFramePresets:
    return _scene_df_presets()
