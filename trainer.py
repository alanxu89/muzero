import time
import copy

import ray
import numpy as np
import tensorflow as tf


import models
from replay_buffer import ReplayBuffer
from shared_storage import SharedStorage
from utils import set_gpu_memory_growth


@ray.remote
class Trainer:
    def __init__(self, config, initial_checkpoint):
        set_gpu_memory_growth()

        self.config = config

        np.random.seed(self.config.seed)
        tf.random.set_seed(self.config.seed)

        self.model = models.MuZeroNetwork(self.config)
        initial_weights = copy.deepcopy(initial_checkpoint["weights"])
        if initial_weights:
            self.model.set_weights(initial_weights)

        self.training_step = initial_checkpoint["training_step"]

        self.lr_schedule = CustomSchedule(
            self.config.lr_init, self.config.lr_decay_rate, self.config.lr_decay_steps)

        self.optimizer = tf.keras.optimizers.Adam(self.lr_schedule
                                                  # weight_decay=self.config.weight_decay,
                                                  )

        if initial_checkpoint["optimizer_state"] is not None:
            self.optimizer.set_weights(copy.deepcopy(
                initial_checkpoint["optimizer_state"]))

    def continuous_update_weights(self,
                                  replay_buffer: ReplayBuffer,
                                  shared_storage: SharedStorage):

        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(1)

        next_batch = replay_buffer.get_batch.remote()

        while self.training_step < self.config.training_steps and not ray.get(shared_storage.get_info.remote("terminate")):
            print("training steps {} and terminate status {}".format(self.training_step,
                                                                     ray.get(shared_storage.get_info.remote("terminate"))))
            batch = replay_buffer.get_batch.remote()
            priorities, total_loss, value_loss, reward_loss, policy_loss = self.update_weights(
                batch)

            if self.config.PER:
                # to be implemented
                # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
                # replay_buffer.update_priorities.remote(priorities, index_batch)
                a = 1

            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info.remote(
                    {
                        "weights": copy.deepcopy(self.model.get_weights()),
                        "optimizer_state": tf.keras.optimizers.serialize(self.optimizer)
                    }
                )
                if self.config.save_model:
                    shared_storage.save_checkpoint.remote()
            shared_storage.set_info.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.get_config()["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            if self.config.training_delay:
                time.sleep(self.config.training_delay)

            if self.config.ratio:
                while(self.training_step /
                      max(1, ray.get(shared_storage.get_info.remote("num_played_steps")))
                      > self.config.ratio and
                      self.training_step < self.config.training_steps
                      and not ray.get(shared_storage.get_info.remote("terminate"))
                      ):
                    time.sleep(0.5)

    def update_weights(self, batch):
        """ train step """
        (
            observation_batch,
            action_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        with tf.GradientTape() as tape:

            target_value_scalar = np.array(target_value, dtype="float32")
            priorities = np.zeros_like(target_value_scalar)
            if self.config.PER:
                weight_batch = tf.Tensor(weight_batch.copy())

            target_value = models.scalar_to_support(
                target_value, self.config.support_size)
            target_reward = models.scalar_to_support(
                target_reward, self.config.support_size
            )
            value, reward, policy_logits, hidden_state = self.model.initial_inference(
                observation_batch
            )
            predictions = [(value, reward, policy_logits)]

            seq_len = action_batch.shape[-1]
            for i in range(1, seq_len):
                value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                    hidden_state, action_batch[:, :, :, i])
                predictions.append((value, reward, policy_logits))

            value_loss, reward_loss, policy_loss = (0, 0, 0)
            value, reward, policy_logits = predictions[0]
            current_value_loss, _, current_policy_loss = self.loss_function(
                value, reward, policy_logits, target_value[0], target_reward[0], target_policy[0])
            value_loss += current_value_loss
            policy_loss += current_policy_loss

            pred_value_scalar = models.support_to_scalar(
                value, self.config.support_size)
            priorities[:, 0] = np.abs(pred_value_scalar -
                                      target_value_scalar[:, 0]) ** self.config.PER_alpha

            for i in range(len(predictions)):
                value, reward, policy_logits = predictions[i]
                current_value_loss, current_reward_loss, current_policy_loss = self.loss_function(
                    value, reward, policy_logits, target_value[:, i], target_reward[:, i], target_policy[:, i])

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i])
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i])
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            pred_value_scalar = models.support_to_scalar(
                value, self.config.support_size)

            priorities[:, 0] = np.abs(pred_value_scalar -
                                      target_value_scalar[:, i]) ** self.config.PER_alpha

            # scale the loss
            loss = value_loss*self.config.value_loss_weight + reward_loss + policy_loss
            if self.config.PER:
                loss *= weight_batch
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_weights))
        self.training_step += 1

    @staticmethod
    def loss_function(
            value,
            reward,
            policy_logits,
            target_value,
            target_reward,
            target_policy):

        value_loss = -tf.reduce_sum(target_value * tf.nn.log_softmax(value))
        reward_loss = -tf.reduce_sum(target_reward * tf.nn.log_softmax(value))
        policy_loss = -tf.reduce_sum(
            target_policy * tf.nn.log_softmax(policy_logits))

        return value_loss, reward_loss, policy_loss


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, lr_init, lr_decay_rate, lr_decay_steps):
        super().__init__()
        self.lr_init = lr_init
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_steps = lr_decay_steps

    def __call__(self, step):
        return self.lr_init * self.lr_decay_rate ** (step / self.lr_decay_steps)
