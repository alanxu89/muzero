import copy
import time
import numpy as np
from numpy.core.defchararray import index
from numpy.lib.function_base import gradient


class ReplayBuffer:
    def __init__(self, config, initial_checkpoint, initial_buffer):
        self.config = config
        self.buffer = copy.deepcopy(initial_buffer)
        self.num_played_games = initial_checkpoint["num_played_games"]
        self.num_played_steps = initial_checkpoint["num_played_steps"]
        self.total_samples = sum([len(game_history.root_values)
                                  for game_history in self.buffer.values()])
        if self.total_samples != 0:
            print(
                f"Replay buffer initialized with {self.total_samples} samples ({self.num_played_games} games).\n"
            )

        np.random.seed(self.config.seed)

    def save_game(self, game_history, shared_storage=False):
        # priority
        if self.config.PER:
            if game_history.priorities is not None:
                # Avoid read only array when loading replay buffer from disk
                game_history.priorities = np.copy(game_history.priorities)
            else:
                priorities = []
                for i, root_value in enumerate(game_history.root_values):
                    priority = (
                        np.abs(
                            root_value -
                            self.compute_target_value(game_history, i)
                        )
                    ) ** self.config.PER_alpha
                priorities.append(priority)

                game_history.priorities = np.array(priorities, dtype="float32")
                game_history.game_priority = np.max(game_history.priorities)

        self.buffer[self.num_played_games] = game_history
        self.num_played_games += 1
        self.num_played_steps += len(game_history.root_values)
        self.total_samples += len(game_history.root_values)

        if len(self.buffer) > self.config.replay_buffer_size:
            del_id = self.num_played_games - len(self.buffer)
            self.total_samples -= len(self.buffer[del_id].root_values)
            del self.buffer[del_id]

    def get_buffer(self):
        return self.buffer

    def get_batch(self):
        index_batch = []
        observation_batch = []
        action_batch = []
        reward_batch = []
        value_batch = []
        policy_batch = []
        gradient_scale_batch = []
        weight_batch = [] if self.config.PER else None

        for game_id, game_history, game_prob in self.sample_n_games(self.config.batch_size):
            game_pos, pos_prob = self.sample_position(game_history)
            values, rewards, policies, actions = self.make_target(
                game_history, game_pos)

            index_batch.append([game_id, game_pos])
            observation_batch.append([
                game_history.get_stacked_observations(
                    game_pos, self.config.stacked_observations)
            ])

            action_batch.append(actions)
            value_batch.append(values)
            reward_batch.append(rewards)
            policy_batch.append(policies)

            gradient_scale_batch.append(
                [min(self.config.unrolled_steps, len(
                    game_history.action_history) - game_pos)]
                * len(actions)
            )

            if self.config.PER:
                weight_batch.append(
                    1.0/(self.total_samples*game_prob*pos_prob))
        return (
            index_batch,
            (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            )
        )

    def sample_game(self, force_uniform=False):
        game_prob = None
        if self.config.PER and not force_uniform:
            game_probs = np.array(
                [game_history.game_priority for game_history in self.buffer.values()])
            game_probs /= np.sum(game_probs)
            game_index = np.random.choice(len(self.buffer), game_probs)
            game_prob = game_probs[game_index]
        else:
            game_index = np.random.choice(len(self.buffer))
        game_id = self.num_played_games - len(self.buffer) + game_index

        return game_id, self.buffer[game_id], game_prob

    def sample_n_games(self, n_games, force_uniform=False):
        if self.config.PER and not force_uniform:
            game_id_list = []
            game_probs = []
            for game_id, game_history in self.buffer.items():
                game_id_list.append(game_id)
                game_probs.append(game_history.game_priority)
            game_probs = np.array(game_probs, dtype="float32")
            game_probs /= np.sum(game_probs)
            game_prob_dict = dict(
                [(game_id, prob) for game_id, prob in zip(game_id_list, game_probs)])
            selected_games = np.random.choice(
                game_id_list, n_games, p=game_probs)
        else:
            selected_games = np.random.choice(
                list(self.buffer.keys()), n_games)
            game_prob_dict = {}
        ret = [(game_id, self.buffer[game_id], game_prob_dict.get(game_id))
               for game_id in selected_games]
        return ret

    def sample_position(self, game_history, force_uniform=False):
        position_prob = None
        if self.config.PER and not force_uniform:
            position_probs = game_history.priorities / \
                sum(game_history.priorities)
            position_index = np.random.choice(
                len(position_probs), p=position_probs)
            position_prob = position_probs[position_index]
        else:
            position_index = numpy.random.choice(len(game_history.root_values))

        return position_index, position_prob

    def update_game_history(self, game_id, game_history):
        if next(iter(self.buffer)) <= game_id:
            if self.config.PER:
                game_history.priorities = np.copy(game_history.priorities)
            self.buffer[game_id] = game_history

    def update_priorities(self, priorities, index_info):
        for i in range(len(index_info)):
            game_id, game_pos = index_info[i]

            if next(iter(self.buffer)) <= game_id:
                priority = priorities[i, :]
                start_index = game_pos
                end_index = min(game_pos + len(priority),
                                len(self.buffer[game_id].priorities))
                self.buffer[game_id].priorities[start_index:
                                                end_index] = priority[:end_index - start_index]

                self.buffer[game_id].game_priority = np.max(
                    self.buffer[game_id].priorities)

    def compute_target_value(self, game_history, index):
        bootstrap_index = index + self.config.td_steps
        if bootstrap_index < len(game_history.root_values):
            root_values = game_history.root_values
            last_step_value = (
                root_values[bootstrap_index] if game_history.to_play_history[
                    bootstrap_index] == game_history.to_play_history[index] else -root_values[bootstrap_index]
            )
            value = last_step_value * \
                (self.config.discount ** self.config.td_steps)
        else:
            value = 0

        for i, reward in enumerate(game_history.reward_history[index+1: bootstrap_index + 1]):
            # account for two-player game
            value += (reward if game_history.to_play_history[index]
                      == game_history.to_play_history[index + 1] else -reward) * (self.config.discount ** i)

        return value

    def make_target(self, game_history, state_index):
        """ Generate targets for every unroll steps """
        target_values = []
        target_rewards = []
        target_policies = []
        actions = []

        for current_index in range(state_index, state_index + self.config.num_unroll_steps+1):
            value = self.compute_target_value(game_history, current_index)

            if current_index < len(game_history.root_values):
                target_values.append(value)
                target_rewards.append(
                    game_history.reward_history[current_index])
                target_policies.append(
                    game_history.child_visits[current_index])
                actions.append(game_history.action_history[current_index])
            elif current_index == len(game_history.root_values):
                target_values.append(0)
                target_rewards.append(
                    game_history.reward_history[current_index])
                # uniform policy
                n = len(game_history.child_visits[0])
                target_policies.append([1.0 / n] * n)
                actions.append(game_history.action_history[current_index])
            else:
                target_values.append(0)
                target_rewards.append(0)
                # Uniform policy
                n = len(game_history.child_visits[0])
                target_policies.append([1.0 / n] * n)
                actions.append(np.random.choice(self.config.action_space))

        return target_values, target_rewards, target_policies, actions
