from mlagents.plugins import all_trainer_types, all_trainer_settings
from mlagents.trainers.dqn.dqn_trainer import DQNTrainer
from mlagents.trainers.dqn.dqn_settings import DQNSettings

# Register DQN trainer
all_trainer_types["dqn"] = DQNTrainer
all_trainer_settings["dqn"] = DQNSettings

# Test YAML config parsing
trainer_config = {
    "trainer_type": "dqn",
    "hyperparameters": {
        "epsilon_start": 1.0,
        "epsilon_end": 0.1,
        "epsilon_decay": 1000,
        "learning_rate": 0.001,
        "gamma": 0.99,
        "batch_size": 64,
        "replay_buffer_size": 10000,
        "hidden_units": 128,
        "update_frequency": 4,
    },
}

# Simulate TrainerSettings.structure
from mlagents.trainers.settings import TrainerSettings
trainer_settings = TrainerSettings.structure(trainer_config, TrainerSettings)
print("DEBUG: Parsed trainer settings:", trainer_settings)
