from dataclasses import dataclass

from dronalize.datasets import list_datasets


@dataclass(slots=True)
class DatasetCase:
    dataset: str
    path_rel_root: str
    max_scenes: int = 5
    scene_start: int = 0
    scene_step: int = 100


ALL_CASES_DEFAULT: dict[str, DatasetCase] = {
    name: DatasetCase(name, path_rel_root=name) for name in list_datasets()
}
