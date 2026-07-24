import unittest

import numpy as np

from rl.memory import ReplayMemory
from rl.agent import TrafficSignalAgent
from rl.reward import compute_reward


class TestRLComponents(unittest.TestCase):

    def test_replay_memory(self):
        memory = ReplayMemory(capacity=5)
        memory.push([0.0], 0, 1.0, [0.1], False)
        memory.push([0.1], 1, 0.0, [0.2], True)
        states, actions, rewards, next_states, dones = memory.sample(2)
        self.assertEqual(states.shape, (2, 1))
        self.assertEqual(actions.shape, (2,))
        self.assertEqual(rewards.shape, (2,))

    def test_agent_interface(self):
        agent = TrafficSignalAgent(
            state_dim=4,
            action_dim=3,
            memory_capacity=10,
            gamma=0.9,
            learning_rate=1e-3,
            target_update_frequency=5,
        )
        state = np.zeros(4, dtype=np.float32)
        action = agent.select_action(state, epsilon=0.0)
        self.assertIn(action, [0, 1, 2])
        agent.remember(state, action, 1.0, state, False)
        loss = agent.update(batch_size=2)
        self.assertEqual(loss, 0.0)

    def test_reward_function(self):
        previous = {
            "queue_length": 10,
            "downstream_occupancy": 0.8,
            "traffic_event_type": 2,
        }
        current = {
            "queue_length": 8,
            "downstream_occupancy": 0.5,
            "traffic_event_type": 1,
        }
        reward = compute_reward(previous, current, action=0, action_executed=True)
        self.assertGreater(reward, 0)


if __name__ == "__main__":
    unittest.main()
