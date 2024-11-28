from dataclasses import dataclass
from mlagents.trainers.settings import TrainerSettings

@dataclass
class DQNSettings(TrainerSettings):
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay: int = 1000
    learning_rate: float = 0.001
    gamma: float = 0.99
    batch_size: int = 64
    replay_buffer_size: int = 10000
    hidden_units: int = 128
    update_frequency: int = 4
