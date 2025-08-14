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

"""Motion models for neural differential equations.

This module provides various motion models including first-order and second-order
grey neural ODEs, as well as single and double integrator models.
"""

import torch
from torch import nn

from models.mtp_go.layers.solvers import solvers


def select_mm(model: str) -> type[nn.Module]:
    """Select motion model class by name.

    Args:
        model: Name of the motion model.

    Returns:
        Motion model class.

    Raises:
        ValueError: If model name is not recognized.

    """
    models = {
        "FirstOrderGreyNeuralODE": FirstOrderGreyNeuralODE,
        "SecondOrderGreyNeuralODE": SecondOrderGreyNeuralODE,
        "SingleIntegrator": SingleIntegrator,
        "DoubleIntegrator": DoubleIntegrator,
    }

    if model not in models:
        msg = f"Model {model} not found."
        raise ValueError(msg)

    return models[model]


class MotionModelBase(nn.Module):
    """Base class for all motion models.

    This class provides common functionality for motion models including
    solver integration, input constraints, and state transitions.

    Args:
        solver: Solver method name (default: "rk4").
        dt: Time step size (default: 4e-2).
        n_states: Number of state variables (default: 4).
        mixtures: Number of mixture components (default: 6).
        *u_lims: Variable number of input limits.

    """

    def __init__(
        self,
        solver: str = "rk4",
        dt: float = 4e-2,
        n_states: int = 4,
        mixtures: int = 6,
        *u_lims: float,
    ) -> None:
        super().__init__()
        self.dt = dt
        self.mixtures = mixtures
        self.n_states = n_states
        self.t_states = n_states
        self.n_inputs = len(u_lims)

        # Pre-compute constraint functions for better performance
        self.u_constrain = nn.ModuleList(
            [nn.Hardtanh(-u_lim, u_lim) for u_lim in u_lims]
        )

        self.solver = solvers[solver]
        self.jac_solver = solvers["ef"]

        # Pre-compute input transition matrix
        self.G = nn.Parameter(self._build_inp_transition_matrix(), requires_grad=False)

    def _build_inp_transition_matrix(self) -> torch.Tensor:
        """Build the input transition matrix G.

        Returns:
            Input transition matrix of shape (1, mixtures, n_states, n_inputs).

        """
        G = torch.zeros(1, self.mixtures, self.n_states, self.n_inputs)
        idx_offset = self.n_states - self.n_inputs

        # Set diagonal elements for input transition
        G[..., idx_offset, 0] = 1.0 * self.dt
        G[..., idx_offset + 1, 1] = 1.0 * self.dt

        return G

    def model_update(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Update the model state (to be implemented by subclasses).

        Args:
            x: Current state tensor.
            u: Input tensor.

        Returns:
            State derivative tensor.

        """
        return x

    def forward(
        self, past_state: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the motion model.

        Args:
            past_state: Previous state tensor.
            inputs: Input tensor.

        Returns:
            Tuple of (next_state, clamped_inputs).

        """
        # Apply input constraints efficiently
        input_clamped = torch.stack(
            [
                constraint(inputs[..., i])
                for i, constraint in enumerate(self.u_constrain)
            ],
            dim=-1,
        )

        next_state = self.solver(self.model_update, past_state, input_clamped, self.dt)

        return next_state, input_clamped

    def state_transition(
        self, state: torch.Tensor, inputs: torch.Tensor
    ) -> torch.Tensor:
        """Compute state transition.

        Args:
            state: Current state tensor.
            inputs: Input tensor.

        Returns:
            Next state tensor.

        """
        return self.jac_solver(self.model_update, state, inputs, self.dt)

    @torch.inference_mode(False)
    def state_transition_matrix(
        self, x: torch.Tensor, inp: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute state transition matrix and its transpose.

        Args:
            x: State tensor of shape (N, m, feat_dim).
            inp: Input tensor.

        Returns:
            Tuple of (F, F^T) where F is the Jacobian matrix.

        """
        N, m, feat_dim = x.shape

        jacobian_rev = torch.vmap(torch.func.jacrev(self.state_transition, argnums=0))(
            x.flatten(0, 1), inp.flatten(0, 1)
        )

        F = jacobian_rev.view(N, m, feat_dim, feat_dim)
        F_T = F.transpose(-2, -1)

        return F, F_T

    def input_transition_matrix(
        self, x: torch.Tensor, *args
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute input transition matrix and its transpose.

        Args:
            x: State tensor.
            *args: Additional arguments (unused).

        Returns:
            Tuple of (G, G^T) where G is the input transition matrix.

        """
        batch_size = x.size(0)
        G = self.G.expand(batch_size, -1, -1, -1)
        G_T = G.transpose(-2, -1)

        return G, G_T


class FirstOrderGreyNeuralODE(MotionModelBase):
    """First-order grey neural ODE model.

    This model uses neural networks to learn the dynamics:
    dx/dt, dy/dt = NN(x, y, u1, u2)

    Args:
        solver: Solver method name (default: "rk4").
        dt: Time step size (default: 4e-2).
        n_states: Number of state variables (default: 2).
        mixtures: Number of mixture components (default: 6).
        n_hidden: Number of hidden units (default: 16).
        n_layers: Number of hidden layers (default: 1).
        u1_lim: Limit for first input (default: 10).
        u2_lim: Limit for second input (default: 10).

    """

    def __init__(
        self,
        solver: str = "rk4",
        dt: float = 4e-2,
        n_states: int = 2,
        mixtures: int = 6,
        n_hidden: int = 16,
        n_layers: int = 1,
        u1_lim: float = 10,
        u2_lim: float = 10,
    ) -> None:
        super().__init__(solver, dt, n_states, mixtures, u1_lim, u2_lim)
        self.n_states = n_states

        # Create separate networks for each state derivative
        self.f = nn.ModuleList(
            [self._create_net(n_states, 1, n_hidden, n_layers) for _ in range(n_states)]
        )

    @staticmethod
    def _create_net(
        n_states: int, n_inputs: int, n_hidden: int, n_layers: int = 2
    ) -> nn.Sequential:
        """Create a neural network for state derivative computation.

        Args:
            n_states: Number of state variables.
            n_inputs: Number of input variables.
            n_hidden: Number of hidden units.
            n_layers: Number of hidden layers.

        Returns:
            Sequential neural network.

        """
        layers = []
        input_size = n_inputs + n_states

        # Hidden layers
        for i in range(n_layers):
            if i == 0:
                layers.extend([nn.Linear(input_size, n_hidden), nn.ELU(inplace=True)])
            else:
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.ELU(inplace=True)])

        # Output layer
        output_size = n_hidden if n_layers > 0 else input_size
        layers.append(nn.Linear(output_size, 1))

        return nn.Sequential(*layers)

    def model_update(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Update model state using neural networks.

        Args:
            x: Current state tensor.
            u: Input tensor.

        Returns:
            State derivative tensor.

        """
        u_x = u[..., 0:1]
        u_y = u[..., 1:2]

        # Concatenate state and respective input for each dimension
        inp_x = torch.cat([x, u_x], dim=-1)
        inp_y = torch.cat([x, u_y], dim=-1)

        # Compute derivatives
        dx = self.f[0](inp_x)
        dy = self.f[1](inp_y)

        return torch.cat([dx, dy], dim=-1)


class SecondOrderGreyNeuralODE(MotionModelBase):
    """Second-order grey neural ODE model.

    This model uses neural networks to learn acceleration dynamics:
    dx/dt = vx, dy/dt = vy
    dvx/dt, dvy/dt = NN(vx, vy, u1, u2)

    Args:
        solver: Solver method name (default: "rk4").
        dt: Time step size (default: 4e-2).
        n_states: Number of state variables (default: 4).
        mixtures: Number of mixture components (default: 6).
        n_hidden: Number of hidden units (default: 16).
        n_layers: Number of hidden layers (default: 1).
        u1_lim: Limit for first input (default: 10).
        u2_lim: Limit for second input (default: 10).

    """

    def __init__(
        self,
        solver: str = "rk4",
        dt: float = 4e-2,
        n_states: int = 4,
        mixtures: int = 6,
        n_hidden: int = 16,
        n_layers: int = 1,
        u1_lim: float = 10,
        u2_lim: float = 10,
    ) -> None:
        super().__init__(solver, dt, n_states, mixtures, u1_lim, u2_lim)
        self.n_states = n_states

        # Neural networks for velocity derivatives
        self.f = nn.ModuleList(
            [
                self._create_net(n_states // 2, 1, n_hidden, n_layers)
                for _ in range(n_states // 2)
            ]
        )

        # Transformation matrices
        self._setup_transformation_matrices()

    def _setup_transformation_matrices(self) -> None:
        """Set up transformation matrices for state and input processing."""
        # Position derivatives (x, y) = (vx, vy)
        self.m0 = nn.Parameter(
            torch.zeros(self.n_states, self.n_states // 2), requires_grad=False
        )
        self.m0[2, 0] = 1.0  # dx/dt = vx
        self.m0[3, 1] = 1.0  # dy/dt = vy

        # Input transformation for vy computation
        self.m1 = nn.Parameter(torch.zeros(6, 3), requires_grad=False)
        self.m1[2, 0] = 1.0  # vx
        self.m1[3, 1] = 1.0  # vy
        self.m1[4, 2] = 1.0  # u2

        # Input transformation for vx computation
        self.m2 = nn.Parameter(torch.zeros(6, 3), requires_grad=False)
        self.m2[2, 0] = 1.0  # vx
        self.m2[3, 1] = 1.0  # vy
        self.m2[5, 2] = 1.0  # u1

    @staticmethod
    def _create_net(
        n_states: int, n_inputs: int, n_hidden: int, n_layers: int = 2
    ) -> nn.Sequential:
        """Create a neural network for acceleration computation.

        Args:
            n_states: Number of state variables.
            n_inputs: Number of input variables.
            n_hidden: Number of hidden units.
            n_layers: Number of hidden layers.

        Returns:
            Sequential neural network.

        """
        layers = []
        input_size = n_inputs + n_states

        # Hidden layers
        for i in range(n_layers):
            if i == 0:
                layers.extend([nn.Linear(input_size, n_hidden), nn.ELU(inplace=True)])
            else:
                layers.extend([nn.Linear(n_hidden, n_hidden), nn.ELU(inplace=True)])

        # Output layer
        output_size = n_hidden if n_layers > 0 else input_size
        layers.append(nn.Linear(output_size, 1))

        return nn.Sequential(*layers)

    def model_update(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Update model state using second-order dynamics.

        Args:
            x: Current state tensor.
            u: Input tensor.

        Returns:
            State derivative tensor.

        """
        inp = torch.cat([x, u], dim=-1)

        # Transform inputs for each velocity component
        inp_vx = inp @ self.m2
        inp_vy = inp @ self.m1

        # Compute accelerations
        dvx = self.f[0](inp_vx)
        dvy = self.f[1](inp_vy)

        # Combine position and velocity derivatives
        return torch.cat([x @ self.m0, dvx, dvy], dim=-1)


class SingleIntegrator(MotionModelBase):
    """Single integrator model.

    Simple kinematic model where velocities are directly controlled:
    dx/dt = u1, dy/dt = u2

    Args:
        solver: Solver method name (default: "ef").
        dt: Time step size (default: 2e-1).
        n_states: Number of state variables (default: 2).
        mixtures: Number of mixture components (default: 6).
        u1_lim: Limit for first input (default: 60).
        u2_lim: Limit for second input (default: 12).

    """

    def __init__(
        self,
        solver: str = "ef",
        dt: float = 2e-1,
        n_states: int = 2,
        mixtures: int = 6,
        u1_lim: float = 60,
        u2_lim: float = 12,
    ) -> None:
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)

        # Identity transformation matrix
        self.m1 = nn.Parameter(torch.eye(n_states), requires_grad=False)

    def model_update(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Update model state using single integrator dynamics.

        Args:
            x: Current state tensor (unused in single integrator).
            u: Input tensor containing velocities.

        Returns:
            State derivative tensor equal to input velocities.

        """
        return u @ self.m1


class DoubleIntegrator(MotionModelBase):
    """Double integrator model.

    Classic double integrator dynamics:
    dx/dt = vx, dy/dt = vy
    dvx/dt = u1, dvy/dt = u2

    Args:
        solver: Solver method name (default: "rk4").
        dt: Time step size (default: 2e-1).
        n_states: Number of state variables (default: 4).
        mixtures: Number of mixture components (default: 6).
        u1_lim: Limit for first input (default: 10).
        u2_lim: Limit for second input (default: 8).

    """

    def __init__(
        self,
        solver: str = "rk4",
        dt: float = 2e-1,
        n_states: int = 4,
        mixtures: int = 6,
        u1_lim: float = 10,
        u2_lim: float = 8,
    ) -> None:
        super().__init__(solver, dt, n_states, mixtures, 0, u1_lim, u2_lim)

        # Position derivative transformation: [x, y] -> [vx, vy]
        self.m0 = nn.Parameter(
            torch.zeros(n_states, n_states // 2), requires_grad=False
        )
        self.m0[2, 0] = 1.0  # dx/dt = vx
        self.m0[3, 1] = 1.0  # dy/dt = vy

        # Acceleration transformation: identity for inputs
        self.m1 = nn.Parameter(torch.eye(n_states // 2), requires_grad=False)

    def model_update(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """Update model state using double integrator dynamics.

        Args:
            x: Current state tensor [x, y, vx, vy].
            u: Input tensor containing accelerations [ax, ay].

        Returns:
            State derivative tensor [vx, vy, ax, ay].

        """
        # Position derivatives from velocities
        position_derivatives = x @ self.m0

        # Velocity derivatives from accelerations
        velocity_derivatives = u @ self.m1

        return torch.cat([position_derivatives, velocity_derivatives], dim=-1)
