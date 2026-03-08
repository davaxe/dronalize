from streaming import MDSWriter as _MDSWriter

from dronalize.core.datatypes.scene import Scene
from dronalize.core.protocols.writer import SceneWriter


class MDSWriter(SceneWriter):
    def __init__(self, output_dir: str) -> None:
        self._inner = _MDSWriter(out=output_dir, columns={"test": "pkl"})

    def write(self, processed: Scene) -> None:
        self._inner.write(sample={"test": processed})

    def finalize(self) -> None:
        self._inner.finish()
