# Copyright 2024-2025, Theodor Westny. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import pickle
import threading
from multiprocessing import Manager, Pool
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from tqdm import tqdm

from preprocessing.arguments import args
from preprocessing.utils import common
from preprocessing.utils.ethucy_utils import (
    PedestrianLoaderConfig,
    PedestrianSampleLoader,
    Split,
)

if TYPE_CHECKING:
    from multiprocessing.queues import Queue


def get_config(
    data_set_name: str,
    split: Split,
    yaml_config: dict[str, Any],
) -> PedestrianLoaderConfig:
    """Get the configuration for the pedestrian dataset loader.

    Args:
        data_set_name: nam of the dataset, loaded from yaml_config.
        split: the split of the dataset, e.g., train, val, test.
        yaml_config: original full yaml configuration file.

    Returns:
        PedestrianDataConfig: Configuration object for the dataset loader.

    """
    split_config = yaml_config[f"{split}_config"]
    return PedestrianLoaderConfig(
        dataset=data_set_name,
        data_root=Path(args.path) / yaml_config.get("dataset", "ethucy"),
        split=split,
        org_sample_time=1 / yaml_config.get("sample_freq", 2.5),
        org_obs_len=yaml_config.get("input_len", 8),
        org_pred_len=yaml_config.get("pred_len", 12),
        interpolation_factor=yaml_config.get("upsample", 1),
        # Split-specific configurations
        min_pedestrian=split_config.get(
            "min_pedestrian",
            1,
        ),
        multiple_targets_per_window=split_config.get(
            "multiple_targets_per_window",
            False,
        ),
        require_all_valid=split_config.get(
            "require_all_valid",
            False,
        ),
    )


def load_dataset(
    data_set_name: str,
    yaml_config: dict[str, Any],
    queue: Queue[int],
) -> None:
    """Load the dataset and save it in the specified format.

    Args:
        data_set_name: name of the dataset, loaded from yaml_config.
        yaml_config: original full yaml configuration file.
        queue: multiprocessing queue to track progress.

    Returns:
        LoadResult: Result of the loading operation, indicating success or failure.

    """
    splits: list[Split] = ["train", "val", "test"]
    for split in splits:
        output_dir = Path(args.output_dir) / data_set_name / split
        output_dir.mkdir(parents=True, exist_ok=True)
        config = get_config(data_set_name, split, yaml_config)
        loader = PedestrianSampleLoader(config)
        for i, agent_sample in enumerate(loader):
            output_file = output_dir / f"{data_set_name}_{split}_{i:05d}.pt"
            data_dict = {"agent": agent_sample} | common.dummy_map()

            if not args.debug:
                pickle.dump(
                    data_dict,
                    Path.open(output_file, "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            queue.put(1)
    queue.put(0)


def monitor_progress(queue: Queue[int], total: int) -> None:
    """Monitor the progress of dataset loading."""
    with tqdm(total=20448) as p_bar:  # 150349
        counter = 0
        while counter < total:
            if not queue.get():
                counter += 1
            else:
                p_bar.update()


def main(yaml_config_path: Path) -> None:
    """Run the dataset loading process."""
    if args.debug:
        print("DEBUG MODE: ON \n")

    with yaml_config_path.open("r") as file:
        config: dict[str, Any] = yaml.safe_load(file)

    with Manager() as manager:
        queue = manager.Queue()

        datasets = [
            name
            for name, dataset_config in config["datasets"].items()
            if dataset_config.get("include", False)
        ]

        monitor = threading.Thread(
            target=monitor_progress,
            daemon=True,
            args=(queue, len(datasets)),
        )

        monitor.start()
        if args.use_threads:
            with Pool() as pool:
                pool.starmap(
                    load_dataset,
                    [(name, config, queue) for name in datasets],
                )

        else:
            for name in datasets:
                load_dataset(name, config, queue)  # type: ignore[reportArgumentType]

        monitor.join()


if __name__ == "__main__":
    config_file = args.config
    if not config_file.endswith(".yml"):
        config_file += ".yml"

    path = "preprocessing/configs/" + config_file
    main(Path(path))
