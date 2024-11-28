from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.dqn.dqn_buffer import ReplayBuffer
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.dqn.policy import DQNPolicy
from mlagents.trainers.dqn.dqn_optimizer import DQNTrainer as DQNTorchTrainer
from mlagents.trainers.dqn.dqn_settings import DQNSettings

TRAINER_NAME = "dqn"

class DQNTrainer(RLTrainer):
    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: DQNSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):
        super().__init__(
            behavior_name,
            trainer_settings,
            training,
            load,
            artifact_path,
            reward_buff_cap,
        )
        self.seed = seed
        self.hyperparameters = trainer_settings.hyperparameters
        self.policy: DQNPolicy = None
        self.optimizer: DQNTorchTrainer = None
        self.update_buffer = ReplayBuffer(capacity=self.hyperparameters.buffer_size)

    def create_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec
    ) -> DQNPolicy:
        """Create a DQN policy."""
        state_dim = behavior_spec.observation_shapes[0][0]
        action_dim = behavior_spec.action_spec.discrete_branches[0]
        self.policy = DQNPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            epsilon=self.hyperparameters.epsilon_start,
            learning_rate=self.hyperparameters.learning_rate,
        )
        return self.policy

    def create_optimizer(self) -> DQNTorchTrainer:
        """Create the optimizer."""
        return DQNTorchTrainer(
            state_dim=self.policy.network.fc1.in_features,
            action_dim=self.policy.network.fc3.out_features,
            epsilon=self.hyperparameters.epsilon_start,
            learning_rate=self.hyperparameters.learning_rate,
            gamma=self.hyperparameters.gamma,
        )

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """Process and add trajectories to the replay buffer."""
        agent_buffer = trajectory.to_agentbuffer()
        for step in agent_buffer:
            self.update_buffer.append_update(
                state=step["obs"][0],
                action=step["action"][0],
                reward=step["reward"],
                next_state=step["next_obs"][0],
                done=step["done"],
            )

    def _is_ready_update(self) -> bool:
        """Check if enough samples are in the replay buffer."""
        return self.update_buffer.num_experiences >= self.hyperparameters.batch_size

    def _update_policy(self) -> bool:
        """Update the DQN policy."""
        if not self._is_ready_update():
            return False

        minibatch = self.update_buffer.sample_mini_batch(
            self.hyperparameters.batch_size
        )
        self.optimizer.train_step(
            minibatch["state"],
            minibatch["action"],
            minibatch["reward"],
            minibatch["next_state"],
            minibatch["done"],
        )
        self.update_buffer.clear()
        return True

    def advance(self) -> None:
        """Advance the trainer by processing trajectories and updating the policy."""
        for traj_queue in self.trajectory_queues:
            while not traj_queue.empty():
                traj = traj_queue.get()
                self._process_trajectory(traj)

        if self.should_still_train and self._is_ready_update():
            self._update_policy()

    def save_model(self) -> None:
        """Save the trained model."""
        self.model_saver.save_checkpoint(self.policy)

    @staticmethod
    def get_trainer_name() -> str:
        return "dqn"
