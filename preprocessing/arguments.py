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


parser = ArgumentParser(description="Preprocessing arguments")

# Program arguments
parser.add_argument(
    "--path",
    "-p",
    type=str,
    default="../datasets",
    help="path to dataset (default: ../datasets)",
)
parser.add_argument(
    "--config",
    "-c",
    type=str,
    default="rounD",
    help='name of config file (default: "rounD")',
)
parser.add_argument(
    "--output-dir",
    "-o",
    type=str,
    default="data",
    help="output directory for processed data (default: data)",
)
parser.add_argument(
    "--add-name",
    "-an",
    type=str,
    default="",
    help="additional string to add to output-dir save name (default: empty string)",
)
parser.add_argument(
    "--add-supp",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="if additional features should be added to the data"
    " (polar or road displacement) (default: False)",
)
parser.add_argument(
    "--use-threads",
    "-ut",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="if multiprocessing should be used (default: False)",
)
parser.add_argument(
    "--debug",
    "-d",
    type=str_to_bool,
    default=False,
    const=True,
    nargs="?",
    help="debugging mode (default: False)",
)

args = parser.parse_args()
