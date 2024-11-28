from mlagents.trainers.dqn.policy import DQNPolicy

class DQNTrainer:
    def __init__(self, state_dim, action_dim, epsilon, learning_rate, gamma=0.99):
        self.policy = DQNPolicy(state_dim, action_dim, epsilon, learning_rate)
        self.gamma = gamma

    def train_step(self, state, action, reward, next_state, done):
        self.policy.update(state, action, reward, next_state, done, self.gamma)