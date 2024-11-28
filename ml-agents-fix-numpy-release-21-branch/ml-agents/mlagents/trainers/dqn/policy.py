import numpy as np
import torch
from torch import nn
from typing import Tuple

class DQNPolicy:
    def __init__(self, state_dim, action_dim, epsilon=0.1, learning_rate=0.001):
        self.network = DQNNetwork(state_dim, action_dim)
        self.optimmizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.epsilon = epsilon
        self.action_dim = action_dim

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        else:
            with torch.no_grad():
                return torch.argmax(self.network(torch.tensor(state, dtype=torch.float32))).item()
            
    def update(self, state, action, reward, next_state, done, gamma=0.99):
        state = torch.tensor(state, dtype=torch.float32)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        reward = torch.tensor(reward, dtype=torch.float32)
        action = torch.tensor(action, dtype=torch.int64)
        
        q_values = self.network(state)
        next_q_values = self.network(next_state)
        q_value = q_values[action]
        next_q_value = reward + gamma * torch.max(next_q_values) * (1 - done)

        loss = (q_value - next_q_value.detach()) ** 2
        self.optimmizer.zero_grad()
        loss.backward()
        self.optimmizer.step()

class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
