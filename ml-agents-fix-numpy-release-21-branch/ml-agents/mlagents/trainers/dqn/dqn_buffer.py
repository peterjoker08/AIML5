import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def append_update(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_mini_batch(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[idx] for idx in indices])
        return {
            "state": np.array(states),
            "action": np.array(actions),
            "reward": np.array(rewards),
            "next_state": np.array(next_states),
            "done": np.array(dones),
        }

    @property
    def num_experiences(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()
