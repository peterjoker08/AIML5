import numpy as np

class DummyEnv:
    def __init__(self):
        self.state = np.random.rand(4)
        self.steps = 0
        self.max_steps = 100

    def reset(self):
        self.steps = 0
        self.state = np.random.rand(4)
        return self.state
    
    def step(self, action):
        reward =1.0 if action == 1 else 0.0
        self.steps += 1
        done = self.steps >= self.max_steps
        self.state = np.random.rand(4)
        return self.state, reward, done