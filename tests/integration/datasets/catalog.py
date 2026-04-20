from dataclasses import dataclass

from dronalize.datasets import available


@dataclass(frozen=True)
class DatasetCase:
    dataset: str
    path_rel_root: str
    max_scenes: int = 10
    scene_start: int = 0
    scene_step: int = 10


ALL_CASES_DEFAULT: dict[str, DatasetCase] = {
    name: DatasetCase(name, path_rel_root=name) for name in available()
}
