"""
Traffic signal agent wrapper for reinforcement learning.
"""

from rl.ddqn import DDQNAgent


class TrafficSignalAgent:

    def __init__(self, state_dim, action_dim, **kwargs):
        self.agent = DDQNAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            **kwargs,
        )

    def select_action(self, state, epsilon=0.0):
        return self.agent.select_action(state, epsilon)

    def remember(self, state, action, reward, next_state, done):
        self.agent.remember(
            state,
            action,
            reward,
            next_state,
            done,
        )

    def update(self, batch_size):
        return self.agent.update(batch_size)

    def save(self, path):
        self.agent.save(path)

    def load(self, path):
        self.agent.load(path)
