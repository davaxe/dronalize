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

from argparse import ArgumentParser, ArgumentTypeError


def str_to_bool(value: bool | str) -> bool:
    """Convert a string or boolean value into a proper boolean for argparse parsing.

    This function allows flexible parsing of boolean flags without using
      `store_true` or `store_false`.

    Args:
        value (bool | str): The input value to interpret as a boolean.

    Returns:
        bool: The interpreted boolean value.

    Raises:
        ArgumentTypeError: If the input value cannot be interpreted as a boolean.

    """
    true_vals = ("yes", "true", "t", "y", "1")
    false_vals = ("no", "false", "f", "n", "0")

    if isinstance(value, bool):
        return value
    if value.lower() in true_vals:
        return True
    if value.lower() in false_vals:
        return False

    msg = "Boolean value expected."
    raise ArgumentTypeError(msg)


parser = ArgumentParser(description="Dronalize learning arguments")


# Program arguments
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="random seed (default: 42)",
)
parser.add_argument(
    "--use-logger",
    "-ul",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="if logger should be used (default: False)",
)
parser.add_argument(
    "--use-cuda",
    "-uc",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="if cuda exists and should be used (default: False)",
)
parser.add_argument(
    "--num-workers",
    "-nw",
    type=int,
    default=1,
    help="number of workers in dataloader (default: 1)",
)
parser.add_argument(
    "--pin-memory",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="if pin memory should be used (default: False)",
)
parser.add_argument(
    "--persistent-workers",
    "-pw",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="if persistent workers should be used (default: False)",
)
parser.add_argument(
    "--store-model",
    "-s",
    type=str_to_bool,
    default=True,
    const=True,
    nargs="?",
    help="if checkpoints should be stored (default: True)",
)
parser.add_argument(
    "--overwrite",
    "-ow",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="overwrite if model exists (default: False)",
)
parser.add_argument(
    "--add-name",
    "-an",
    type=str,
    default="",
    help="additional string to add to save name",
)
parser.add_argument(
    "--dry-run",
    "-dr",
    type=str_to_bool,
    default=True,
    const=True,
    nargs="?",
    help="verify the code and the model (default: True)",
)
parser.add_argument(
    "--small-ds",
    "-sd",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="Use tiny versions of dataset for fast testing (default: False)",
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="example",
    help="config file path for experiment (default: example)",
)
parser.add_argument(
    "--pre-train",
    "-pt",
    type=str,
    default="",
    help="file containing a pre-trained model (default: none)",
)
parser.add_argument(
    "--root",
    "-r",
    type=str,
    default="",
    help='root path for dataset (default: "")',
)

args = parser.parse_args()
