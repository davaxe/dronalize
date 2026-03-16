__dronalize_builtin__ = {
    "datasets": ["ad4che"],
    "optional_dependencies": ["cv2"],
    "extra": "ad4che",
}

from dronalize.datasets import _registry

_registry.ensure_builtin_dependencies(__name__, __dronalize_builtin__)

from dronalize.datasets.ad4che.loader import AD4CHELoader  # noqa: E402
from dronalize.datasets.ad4che.map.builder import AD4CHEMapBuilder  # noqa: E402

__all__ = ["AD4CHELoader", "AD4CHEMapBuilder"]

_registry.register(
    _registry.DatasetDescriptor(
        name="ad4che",
        loader_factory=AD4CHELoader,
        default_config=AD4CHELoader.default_config(),
        default_map_config=AD4CHELoader.default_map_config(),
        has_map=True,
        predefined_splits=[],
    )
)
