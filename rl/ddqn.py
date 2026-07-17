"""
Double Deep Q-Network implementation
for proactive traffic signal control.

The DDQN agent uses:

- Online Q-Network for action selection
- Target Q-Network for target evaluation
- Experience Replay
- Epsilon-Greedy Action Selection
- Gradient Clipping
- Periodic Target Network Synchronization
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rl.memory import ReplayMemory
from rl.model import QNetwork


class DDQNAgent:

    def __init__(
        self,
        state_dim,
        action_dim,
        memory_capacity=10000,
        gamma=0.99,
        learning_rate=1e-4,
        target_update_frequency=10,
        device=None,
    ):
        """
        Initialize the Double Deep Q-Network agent.
        """

        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)

        self.gamma = float(gamma)

        self.target_update_frequency = int(
            target_update_frequency
        )

        self.online_net = QNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            input_dim=self.state_dim,
            output_dim=self.action_dim,
        ).to(self.device)

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.target_net.eval()

        self.memory = ReplayMemory(
            memory_capacity
        )

        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
        )

        self.step_counter = 0

    def _validate_state(
        self,
        state,
    ):
        """
        Validate and normalize the state interface.
        """

        state = np.asarray(
            state,
            dtype=np.float32,
        )

        expected_shape = (
            self.state_dim,
        )

        if state.shape != expected_shape:

            raise ValueError(
                "Invalid DDQN state shape. "
                f"Expected {expected_shape}, "
                f"received {state.shape}."
            )

        return state

    def select_action(
        self,
        state,
        epsilon,
    ):
        """
        Select an action using epsilon-greedy policy.
        """

        state = self._validate_state(
            state
        )

        if epsilon is None:

            epsilon = 0.0

        epsilon = float(
            np.clip(
                epsilon,
                0.0,
                1.0,
            )
        )

        if (
            torch.rand(1).item()
            < epsilon
        ):

            return int(
                torch.randint(
                    low=0,
                    high=self.action_dim,
                    size=(1,),
                ).item()
            )

        state_tensor = (
            torch.from_numpy(
                state
            )
            .unsqueeze(0)
            .to(self.device)
        )

        self.online_net.eval()

        with torch.no_grad():

            q_values = self.online_net(
                state_tensor
            )

        self.online_net.train()

        action = q_values.argmax(
            dim=1
        ).item()

        return int(action)

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        """
        Store a traffic transition
        in experience replay memory.
        """

        state = self._validate_state(
            state
        )

        next_state = self._validate_state(
            next_state
        )

        action = int(action)

        if (
            action < 0
            or action >= self.action_dim
        ):

            raise ValueError(
                "Invalid DDQN action. "
                f"Expected action in range "
                f"[0, {self.action_dim - 1}], "
                f"received {action}."
            )

        self.memory.push(
            state,
            action,
            float(reward),
            next_state,
            bool(done),
        )

    def update(
        self,
        batch_size,
    ):
        """
        Perform one Double DQN optimization step.
        """

        batch_size = int(
            batch_size
        )

        if batch_size <= 0:

            raise ValueError(
                "Batch size must be greater than zero."
            )

        if len(self.memory) < batch_size:

            return 0.0

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = self.memory.sample(
            batch_size
        )

        states = torch.from_numpy(
            states
        ).to(
            self.device,
            dtype=torch.float32,
        )

        actions = torch.from_numpy(
            actions
        ).to(
            self.device,
            dtype=torch.long,
        )

        rewards = torch.from_numpy(
            rewards
        ).to(
            self.device,
            dtype=torch.float32,
        )

        next_states = torch.from_numpy(
            next_states
        ).to(
            self.device,
            dtype=torch.float32,
        )

        dones = torch.from_numpy(
            dones
        ).to(
            self.device,
            dtype=torch.float32,
        )

        self.online_net.train()

        current_q = self.online_net(
            states
        ).gather(
            dim=1,
            index=actions.unsqueeze(1),
        ).squeeze(1)

        with torch.no_grad():

            next_actions = self.online_net(
                next_states
            ).argmax(
                dim=1,
                keepdim=True,
            )

            next_q = self.target_net(
                next_states
            ).gather(
                dim=1,
                index=next_actions,
            ).squeeze(1)

            target_q = rewards + (
                self.gamma
                * next_q
                * (
                    1.0
                    - dones
                )
            )

        loss = F.mse_loss(
            current_q,
            target_q,
        )

        self.optimizer.zero_grad()

        loss.backward()

        nn.utils.clip_grad_norm_(
            self.online_net.parameters(),
            max_norm=1.0,
        )

        self.optimizer.step()

        self.step_counter += 1

        if (
            self.step_counter
            % self.target_update_frequency
            == 0
        ):

            self.target_net.load_state_dict(
                self.online_net.state_dict()
            )

            self.target_net.eval()

        return float(
            loss.item()
        )

    def save(
        self,
        path,
    ):
        """
        Save DDQN model checkpoint.
        """

        checkpoint = {
            "online_state_dict": (
                self.online_net.state_dict()
            ),
            "target_state_dict": (
                self.target_net.state_dict()
            ),
            "optimizer_state_dict": (
                self.optimizer.state_dict()
            ),
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "gamma": self.gamma,
            "target_update_frequency": (
                self.target_update_frequency
            ),
            "step_counter": self.step_counter,
        }

        torch.save(
            checkpoint,
            path,
        )

    def load(
        self,
        path,
    ):
        """
        Load DDQN model checkpoint.
        """

        checkpoint = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        checkpoint_state_dim = int(
            checkpoint.get(
                "state_dim",
                self.state_dim,
            )
        )

        checkpoint_action_dim = int(
            checkpoint["action_dim"]
        )

        if (
            checkpoint_state_dim
            != self.state_dim
        ):

            raise ValueError(
                "DDQN checkpoint state dimension "
                "does not match the current agent. "
                f"Checkpoint: {checkpoint_state_dim}, "
                f"Agent: {self.state_dim}."
            )

        if (
            checkpoint_action_dim
            != self.action_dim
        ):

            raise ValueError(
                "DDQN checkpoint action dimension "
                "does not match the current agent. "
                f"Checkpoint: {checkpoint_action_dim}, "
                f"Agent: {self.action_dim}."
            )

        self.online_net.load_state_dict(
            checkpoint[
                "online_state_dict"
            ]
        )

        self.target_net.load_state_dict(
            checkpoint[
                "target_state_dict"
            ]
        )

        if (
            "optimizer_state_dict"
            in checkpoint
        ):

            self.optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        self.step_counter = int(
            checkpoint.get(
                "step_counter",
                0,
            )
        )

        self.online_net.train()

        self.target_net.eval()