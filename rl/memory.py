"""
Replay memory for Deep Q-Learning.
"""

import random
from collections import deque, namedtuple

import numpy as np

Transition = namedtuple(
    "Transition",
    ["state", "action", "reward", "next_state", "done"],
)


class ReplayMemory:

    def __init__(self, capacity):
        if capacity <= 0:
            raise ValueError(
                "ReplayMemory capacity must be greater than zero."
            )

        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            Transition(
                state,
                action,
                reward,
                next_state,
                done,
            )
        )

    def sample(self, batch_size):
        if batch_size <= 0:
            raise ValueError(
                "Batch size must be greater than zero."
            )

        if batch_size > len(self.memory):
            raise ValueError(
                "Requested batch size is larger than memory size."
            )

        transitions = random.sample(
            self.memory,
            batch_size,
        )

        batch = Transition(*zip(*transitions))

        states = np.asarray(
            batch.state,
            dtype=np.float32,
        )

        actions = np.asarray(
            batch.action,
            dtype=np.int64,
        )

        rewards = np.asarray(
            batch.reward,
            dtype=np.float32,
        )

        next_states = np.asarray(
            batch.next_state,
            dtype=np.float32,
        )

        dones = np.asarray(
            batch.done,
            dtype=np.float32,
        )

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
