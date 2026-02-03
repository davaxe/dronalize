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


import importlib
from collections.abc import Callable
from pathlib import Path

import yaml


def load_config(config: str) -> dict:
    """Load a YAML configuration file from the 'configs' directory or its subdirectories.

    Args:
        config (str): The name of the config file (with or without '.yml' extension).

    Returns:
        dict: Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If the specified configuration file is not found in any subdirectory of 'configs'.

    """
    if not config.endswith(".yml"):
        config += ".yml"

    config_path = Path("configs")
    subdirs = [d for d in config_path.iterdir() if d.is_dir()]
    files = [f for d in subdirs for f in d.iterdir() if f.is_file()]

    if not any(config in f.name for f in files):
        msg = f"Config file {config} not found."
        raise FileNotFoundError(msg)

    config = next(str(f) for f in files if config in f.name)

    with open(config, "r", encoding="utf-8") as openfile:
        return yaml.safe_load(openfile)


def import_module(module_name: str) -> object:
    """Dynamically import a module by name.

    Args:
        module_name (str): The name of the module to import.

    Returns:
        object: The imported module object.

    """
    return importlib.import_module(module_name)


def import_from_module(module_name: str, class_name: str) -> Callable:
    """Import a class or callable object from a specified module.

    Args:
        module_name (str): The name of the module.
        class_name (str): The name of the class or callable within the module.

    Returns:
        Callable: The imported class or callable.

    Raises:
        AttributeError: If the specified class_name does not exist in the module.

    """
    module = import_module(module_name)
    return getattr(module, class_name)
