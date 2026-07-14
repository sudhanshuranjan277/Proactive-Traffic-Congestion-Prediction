"""
Double Deep Q-Network implementation for traffic signal control.
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
        self.device = (
            device
            if device is not None
            else torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.action_dim = action_dim

        self.online_net = QNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
        ).to(self.device)

        self.target_net = QNetwork(
            input_dim=state_dim,
            output_dim=action_dim,
        ).to(self.device)

        self.target_net.load_state_dict(
            self.online_net.state_dict()
        )

        self.memory = ReplayMemory(memory_capacity)

        self.gamma = gamma
        self.target_update_frequency = target_update_frequency
        self.optimizer = torch.optim.Adam(
            self.online_net.parameters(),
            lr=learning_rate,
        )

        self.step_counter = 0

    def select_action(self, state, epsilon):
        if epsilon is None or epsilon < 0.0:
            epsilon = 0.0

        if torch.rand(1).item() < epsilon:
            return int(
                torch.randint(
                    0,
                    self.action_dim,
                    (1,),
                ).item()
            )

        state_tensor = torch.from_numpy(
            state.astype(np.float32)
        ).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.online_net(state_tensor)

        return int(q_values.argmax(dim=1).item())

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done,
    ):
        self.memory.push(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return 0.0

        states, actions, rewards, next_states, dones = (
            self.memory.sample(batch_size)
        )

        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)

        current_q = self.online_net(states).gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        next_actions = self.online_net(
            next_states
        ).argmax(dim=1, keepdim=True)

        next_q = self.target_net(
            next_states
        ).gather(
            1,
            next_actions,
        ).squeeze(1)

        target_q = rewards + (
            self.gamma * next_q * (1.0 - dones)
        )

        loss = F.mse_loss(
            current_q,
            target_q.detach(),
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

        return float(loss.item())

    def save(self, path):
        torch.save(
            {
                "online_state_dict": (
                    self.online_net.state_dict()
                ),
                "target_state_dict": (
                    self.target_net.state_dict()
                ),
                "action_dim": self.action_dim,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(
            path,
            map_location=self.device,
        )

        self.online_net.load_state_dict(
            checkpoint["online_state_dict"]
        )

        self.target_net.load_state_dict(
            checkpoint["target_state_dict"]
        )

        self.action_dim = checkpoint["action_dim"]
