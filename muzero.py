import copy
import os
import datetime
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

import ray
import pickle
import numpy as np
import tensorflow as tf

from trainer import Trainer
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from self_play import SelfPlay
from models import MuZeroNetwork


class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        self.observation_shape = (96, 96, 3)
        self.action_space = list(range(18))
        self.players = list(range(1))
        self.stacked_observations = 32

        self.muzero_player = 0
        self.opponent = None

        self.num_workers = 1
        self.selfplay_on_gpu = False
        self.max_moves = 27000
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Value and reward are scaled (with almost sqrt) and encoded on a vector
        # with a range of -support_size to support_size.
        # Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 300

        # Residual Network
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [256, 256]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [256, 256]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [256, 256]

        # Training
        #  Path to store the model weights and TensorBoard logs
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "../results",
                                         os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                                         )
        self.save_model = True
        self.training_steps = int(1e6)
        self.batch_size = 1024
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = int(1e3)
        # Scale the value loss to avoid overfitting of the value function,
        # paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "SGD"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.05
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = int(350e3)

        # Replay Buffer
        self.replay_buffer_size = int(1e6)
        self.num_unroll_steps = 5
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 10
        # Prioritized Replay (See paper appendix Training), select in priority the elements
        # in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 1

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game_name
        self.training_delay = 0  # Number of seconds to wait after each training step
        # Desired training steps per self played step ratio. Equivalent to a synchronous version,
        # training can take much longer. Set it to None to disable it
        self.ratio = None

    def visit_softmax_temperature_fn(self, steps):
        if steps < 500e3:
            return 1.0
        elif steps < 750e3:
            return 0.5
        else:
            return 0.25


class MuZero:
    def __init__(self, game_name="space_invaders", config=None):
        self.game_name = game_name
        self.config = MuZeroConfig()

        if config:
            if type(config) is dict:
                for key, value in config.items():
                    setattr(self.config, key, value)
            else:
                self.config = config

        self.num_gpus = len(tf.config.list_physical_devices('GPU'))
        ray.init(num_gpus=self.num_gpus, ignore_reinit_error=True)

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

        # Checkpoint and replay buffer used to initialize workers
        self.checkpoint = {
            "weights": None,
            "optimizer_state": None,
            "total_reward": 0,
            "muzero_reward": 0,
            "opponent_reward": 0,
            "episode_length": 0,
            "mean_value": 0,
            "training_step": 0,
            "lr": 0,
            "total_loss": 0,
            "value_loss": 0,
            "reward_loss": 0,
            "policy_loss": 0,
            "num_played_games": 0,
            "num_played_steps": 0,
            "num_reanalysed_games": 0,
            "terminate": False,
        }

        self.replay_buffer = {}

        cpu_actor = CPUActor.remote()
        cpu_weights = cpu_actor.get_initial_weights.remote(self.config)
        self.checkpoint["weights"], self.summary = copy.deepcopy(
            ray.get(cpu_weights))

        # Workers
        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def train(self, log_in_tensorboard=True):
        if log_in_tensorboard or self.config.save_model:
            os.makedirs(self.config.results_path, exist_ok=True)

        self.training_worker = Trainer.options(
            num_cpus=0, num_gpus=1).remote(self.config, self.checkpoint)

        self.shared_storage_worker = SharedStorage.remote(
            self.config, self.checkpoint)
        self.shared_storage_worker.set_info.remote("terminate", False)

        self.replay_buffer_worker = ReplayBuffer.remote(
            self.config, self.checkpoint, self.replay_buffer)

        self.self_play_workers = [SelfPlay.options(num_cpus=1, num_gpus=0).remote(self.checkpoint, self.game_name,
                                                                                  self.config, self.config.seed + seed)
                                  for seed in range(self.config.num_workers)]

        # launch self play
        for self_play_worker in self.self_play_workers:
            self_play_worker.continuous_self_play.remote(
                self.shared_storage_worker, self.replay_buffer_worker)

        self.training_worker.continuous_update_weights.remote(
            self.replay_buffer_worker, self.shared_storage_worker)

        if log_in_tensorboard:
            self.logging_loop()

    def logging_loop(self):
        self.test_worker = SelfPlay.remote(
            self.checkpoint, self.game_name, self.config, self.config.seed + self.config.num_workers)
        self.test_worker.continuous_self_play.remote(
            self.shared_storage_worker, None, True)

        hyper_params_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        counter = 0

        with tf.device('/CPU'):
            writer = tf.summary.create_file_writer("/tmp/mylogs/eager")
            with writer.as_default():
                tf.summary.text(
                    "Hyperparameters",
                    "| Parameter | Value |\n|-------|-------|\n" +
                    "\n".join(hyper_params_table), counter
                )

                tf.summary.text("model summary", "", counter)

        keys = [
            "total_reward",
            "muzero_reward",
            "opponent_reward",
            "episode_length",
            "mean_value",
            "training_step",
            "lr",
            "total_loss",
            "value_loss",
            "reward_loss",
            "policy_loss",
            "num_played_games",
            "num_played_steps",
            "num_reanalysed_games",
        ]
        info = ray.get(self.shared_storage_worker.get_info.remote(keys))
        try:
            while info["training_step"] < self.config.training_steps:
                info = ray.get(
                    self.shared_storage_worker.get_info.remote(keys))
                # details to be implemented
                with tf.device('/CPU'):
                    with writer.as_default():
                        tf.summary.scalar("3.Loss/Reward_loss",
                                          info["reward_loss"], counter)
                counter += 1
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass

        if self.config.save_model:
            pickle.dump(
                {
                    "buffer": self.replay_buffer,
                    "num_played_games": self.checkpoint["num_played_games"],
                    "num_played_steps": self.checkpoint["num_played_steps"],
                },
                open(os.path.join(self.config.results_path,
                                  "replay_buffer.pkl"), "wb"),
            )

    def terminate_workers(self):
        print("\nShutting down workers...")

        self.self_play_workers = None
        self.test_worker = None
        self.training_worker = None
        self.reanalyse_worker = None
        self.replay_buffer_worker = None
        self.shared_storage_worker = None

    def test(self, render=True, opponent=None, muzero_player=None, num_tests=1):
        opponent = opponent if opponent else self.config.opponent
        muzero_player = muzero_player if muzero_player else self.config.muzero_player
        self_player_worker = SelfPlay(
            self.checkpoint, self.game_name, self.config, np.random.randint(10000))
        results = []

        for i in range(num_tests):
            print(f"Testing {i+1}/{num_tests}")
            results.append(self_player_worker.play_game(
                0, 0, render, opponent, muzero_player))
        self_player_worker.close_game()

        if len(self.config.players) == 1:
            result = np.mean([sum(history.reward_history)
                              for history in results])
        else:
            result = np.mean([sum(reward for i, reward in enumerate(history.reward_history)
                                  if history.to_play_history[i-1] == muzero_player)
                              for history in results])

    def load_model(self, checkpoint_path=None, replay_buffer_path=None):
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                self.checkpoint = tf.keras.models.load_model(checkpoint_path)
                print(f"\nUsing checkpoint from {checkpoint_path}")
            else:
                print(f"\nThere is no model saved in {checkpoint_path}.")

        if replay_buffer_path:
            if os.path.exists(replay_buffer_path):
                with open(replay_buffer_path, "rb") as f:
                    replay_buffer_infos = pickle.load(f)
                self.replay_buffer = replay_buffer_infos["replay_buffer"]
                self.checkpoint["num_played_steps"] = replay_buffer_infos[
                    "num_played_steps"
                ]
                self.checkpoint["num_played_games"] = replay_buffer_infos[
                    "num_played_games"
                ]
                self.checkpoint["num_reanalysed_games"] = replay_buffer_infos[
                    "num_reanalysed_games"
                ]

                print(
                    f"\nInitializing replay buffer with {replay_buffer_path}")
            else:
                print(
                    f"Warning: Replay buffer path '{replay_buffer_path}' doesn't exist.  Using empty buffer."
                )
                self.checkpoint["training_step"] = 0
                self.checkpoint["num_played_steps"] = 0
                self.checkpoint["num_played_games"] = 0
                self.checkpoint["num_reanalysed_games"] = 0


@ray.remote(num_cpus=1, num_gpus=0)
class CPUActor:
    def __init__(self):
        pass

    def get_initial_weights(self, config):
        model = MuZeroNetwork(config)
        weights = model.get_weights()
        summary = model.get_summary()
        return weights, summary


if __name__ == "__main__":
    muzero = MuZero()
    muzero.train()
