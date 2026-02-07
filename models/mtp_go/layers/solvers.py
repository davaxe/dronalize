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

from collections.abc import Callable

import torch


def forward_euler(
    model: Callable, x: torch.Tensor, u: torch.Tensor, h: float
) -> torch.Tensor:
    """Forward Euler (first-order) method."""
    return x + h * model(x, u)


def midpoint(
    model: Callable, x: torch.Tensor, u: torch.Tensor, h: float
) -> torch.Tensor:
    """Explicit midpoint (second-order) method."""
    k1 = model(x, u)
    k2 = model(x + h / 2 * k1, u)
    return x + h * k2


def heun(model: Callable, x: torch.Tensor, u: torch.Tensor, h: float) -> torch.Tensor:
    """Heun's (second-order) method."""
    k1 = model(x, u)
    k2 = model(x + h * k1, u)
    return x + h / 2 * (k1 + k2)


def rk3(model: Callable, x: torch.Tensor, u: torch.Tensor, h: float) -> torch.Tensor:
    """Kutta's third-order method."""
    k1 = model(x, u)
    k2 = model(x + h / 2 * k1, u)
    k3 = model(x + h * (2 * k2 - k1), u)
    return x + h / 6 * (k1 + 4 * k2 + k3)


def ssprk3(model: Callable, x: torch.Tensor, u: torch.Tensor, h: float) -> torch.Tensor:
    """Third-order Strong Stability Preserving Runge-Kutta (SSPRK3)."""
    k1 = model(x, u)
    k2 = model(x + h * k1, u)
    k3 = model(x + h / 4 * (k1 + k2), u)
    return x + h / 6 * (k1 + k2 + 4 * k3)


def rk4(model: Callable, x: torch.Tensor, u: torch.Tensor, h: float) -> torch.Tensor:
    """Classic fourth-order method."""
    k1 = model(x, u)
    k2 = model(x + h / 2 * k1, u)
    k3 = model(x + h / 2 * k2, u)
    k4 = model(x + h * k3, u)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


solvers = {
    "ef": forward_euler,
    "mp": midpoint,
    "heun": heun,
    "rk3": rk3,
    "ssprk3": ssprk3,
    "rk4": rk4,
}
