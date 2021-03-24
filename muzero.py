import os
import time
import numpy as np
import pickle

import tensorflow as tf

from models import MuZeroResidualNetwork
from self_play import SelfPlay
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from trainer import Trainer
from games.atari import AtariGame, MuZeroConfig


class MuZero:
    def __init__(self, game_name, config):
        self.Game = AtariGame()
        self.config = MuZeroConfig()

        if config:
            if type(config) is dict:
                for key, value in config.items():
                    setattr(self.config, key, value)
            else:
                self.config = config

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

        # Initialize workers
        self.training_worker = Trainer(self.config, self.checkpoint)
        self.shared_storage_worker = SharedStorage(
            self.config, self.checkpoint)
        self.replay_buffer_worker = ReplayBuffer(
            self.config, self.checkpoint, self.replay_buffer)
        self.self_play_workers = [SelfPlay(self.checkpoint, self.Game,
                                           self.config, self.config.seed + seed)
                                  for seed in range(self.config.num_workers)]

        # launch self play
        for self_play_worker in self.self_play_workers:
            self_play_worker.continuous_self_play(
                self.shared_storage_worker, self.replay_buffer_worker)

        self.training_worker.continuous_update_weights(
            self.replay_buffer_worker, self.shared_storage_worker)

        if log_in_tensorboard:
            self.logging_loop()

    def logging_loop(self):
        self.test_worker = SelfPlay(
            self.checkpoint, self.Game, self.config, self.config.seed + self.config.num_workers)
        self.test_worker.continuous_self_play(
            self.shared_storage_worker, None, True)
        writer = tf.summary.create_file_writer("/tmp/mylogs/eager")

        hyper_params_table = [
            f"| {key} | {value} |" for key, value in self.config.__dict__.items()
        ]
        writer.add_text("model summary", "")

        count = 0
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

        try:
            while count < self.config.training_steps:
                # details to be implemented
                writer.add_scalar("3.Loss/Reward_loss", 123, count)
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
