import pytest

from dronalize.core.scene import Scene
from tests.support import make_scene


@pytest.fixture
def scene() -> Scene:
    return make_scene()
