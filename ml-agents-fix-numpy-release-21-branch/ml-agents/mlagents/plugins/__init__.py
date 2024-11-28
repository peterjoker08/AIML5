from typing import Dict, Any
from mlagents.trainers.dqn.dqn_settings import DQNSettings
from mlagents.trainers.dqn.dqn_trainer import DQNTrainer

ML_AGENTS_STATS_WRITER = "mlagents.stats_writer"
ML_AGENTS_TRAINER_TYPE = "mlagents.trainer_type"

# TODO: the real type is Dict[str, HyperparamSettings]
all_trainer_types: Dict[str, Any] = {}
all_trainer_settings: Dict[str, Any] = {}
all_trainer_types['dqn'] = DQNTrainer
all_trainer_settings['dqn'] = DQNSettings
