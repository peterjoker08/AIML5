# test_dqn_learning.py
import unittest
import numpy as np
import matplotlib.pyplot as plt
from mlagents.trainers.dqn.policy import DQNPolicy
from dummy_env import DummyEnv

class TestDQNLearning(unittest.TestCase):
    def test_learning_progress(self):
        # Initialize the environment and the DQN agent
        env = DummyEnv()
        agent = DQNPolicy(state_dim=4, action_dim=2, epsilon=0.1, learning_rate=1e-3)
        episodes = 1000
        rewards = []

        # Training loop
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            done = False

            while not done:
                action = agent.select_action(state)
                next_state, reward, done = env.step(action)
                agent.update(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

            rewards.append(total_reward)

        # Plot the rewards to check if they increase over time
        plt.plot(rewards)
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title("DQN Training Progress")
        plt.show()

        # Check if the average reward in the last 10 episodes is higher than in the first 10
        initial_avg_reward = np.mean(rewards[:10])
        final_avg_reward = np.mean(rewards[-10:])
        self.assertGreater(final_avg_reward, initial_avg_reward, "Agent did not learn; final rewards are not higher than initial rewards")

if __name__ == '__main__':
    unittest.main()
