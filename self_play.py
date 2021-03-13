import numpy as np
import tensorflow as tf

import models


class SelfPlay:
    def __init__(self, initial_checkpoint, Game, config, seed):
        self.config = config
        self.game = Game(seed)

        np.random.seed(seed)
        tf.random.set_seed(seed)

        self.model = models.MuZeroResidualNetwork(
            config.action_space_size, config.num_channels, config.support_size)
        self.model.load_weights(initial_checkpoint)

    def continuous_self_play(self):

    def play_game(self, temperature, temperature_threshold, render, opponent, muzero_player):
        game_history = GameHistory()
        observation = self.game.reset()
        game_history.action_history.append(0)
        game_history.observation_history.append(observation)
        game_history.reward_history.append(0)
        game_history.to_play_history.append(self.game.to_play())

        done = False

        if render:
            self.game.render()

        while not done and len(game_history.action_history) <= self.config.max_moves:
            assert len(numpy.array(
                observation).shape) == 3, "observation should be 3 dimensionnal"
            assert numpy.array(observation).shape ==
                    self.config.observation_shape), "Observation should match the observation_shape defined in MuZeroConfig"


            stacked_observations=game_history.get_stacked_observations(
                -1, self.config.stacked_observations)
            
            if opponent == "self" or muzero_player == self.game.to_play():
                root, extra_info = MCTS(self.config).run(
                    self.model, stacked_observations, self.game.legal_actions(), self.game.to_play(), True
                )
                if (not temperature_threshold) or (len(game_history.action_history) < temperature_threshold):
                    temperature = 0.0
                action = self.select_action(root, temperature)

                if render:
                    print(f'Tree depth: {mcts_info["max_tree_depth"]}')
                    print(f"Root value for player {self.game.to_play()}: {root.value():.2f}")

            else:
                action, root = select_opponent_action(opponent, stacked_observations)

            observation, reward, done = self.game.step(action)
            
            if render:
                    print(f"Played action: {self.game.action_to_string(action)}")
                    self.game.render()

            game_history.store_search_statistics(root, self.config.action_space)

            game_history.action_history.append(action)
            game_history.observation_history.append(observation)
            game_history.reward_history.append(reward)
            game_history.to_play_history.append(self.game.to_play())

        return game_history

    def close_game(self):
        self.game.close()

    def select_opponent_action(self, opponent, stacked_observations):
        """
        Select opponent action for evaluating MuZero level.
        """
        if opponent == "human":
            # true in the last argument
            root, mcts_info = MCTS(self.config).run(
                self.model,
                stacked_observations,
                self.game.legal_actions(),
                self.game.to_play(),
                True,
            )
            return self.game.human_to_action(), root
        elif opponent == "expert":
            return self.game.expert_agent(), None
        elif opponent == "random":
            return numpy.random.choice(self.game.legal_actions()), None
        else:
            raise NotImplementedError(
                'Wrong argument: "opponent" argument should be "self", "human", "expert" or "random"'
            )

    @static_method
    def select_action(node, temperature):
        """
        Select action according to the visit count distribution and the temperature.
        The temperature is changed dynamically with the visit_softmax_temperature function
        in the config.
        """
        visit_counts = np.array([child.visit_count for child in node.children.values()], dtype="int32")
        actions = node.children.keys()

        if temperature < 1e-7:
            np.argmax(visit_counts)]
        elif temperature > 1e7:
            action = np.random.choice(actions)
        else:
            visit_count_distribution = visit_count ** (1.0/ temperature)
            visit_count_distribution = visit_count_distribution/ sum(visit_count_distribution)
            action = np.random.choice(actions, p=visit_count_distribution)
        
        return action



class GameHistory:
    def __init__(self):
        self.observation_history=[]
        self.action_history=[]
        self.reward_history=[]
        self.to_play_history=[]
        self.child_visits=[]
        self.root_values=[]

    def store_search_statistics(self, root, action_space):
        if root is not None:
            sum_visits=sum(
                child.visit_count for child in root.children.values())
            self.child_visits.append([
                root.children[a].visit_count / sum_visits if a in root.children else 0 for a in action_space
            ])
            self.root_values.append(root.value())
        else:
            self.root_values.append(None)

    def get_stacked_observations(self, index, num_stacked_observations):
        index=index % len(self.observation_history)

        stacked_observations=self.observation_history[index].copy()
        for past_observation_index in reversed(range(index - num_stacked_observations, index)):
            if past_observation_index >= 0:
                previous_observation=np.concatenate(
                    [self.observation_history[past_observation_index],
                        np.ones_like(stacked_observations[0])
                        * self.action_history[past_observation_index + 1]
                     ])
            else:
                previous_observation=np.concatenate([
                    np.zeros_like(self.observation_history[index]),
                    np.zeros_like(stacked_observations[0])
                ])

            stacked_observations=np.concatenate(
                [stacked_observations, previous_observation])

        return stacked_observations


class Node:
    def __init__(self, prior):
        self.visit_count=0
        self.to_play=-1
        self.prior=prior
        self.value_sum=0
        self.children={}
        self.hidden_state=None
        self.reward=0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        else:
            return self.reward / self.visit_count

    def expand(self, actions, to_play, reward, policy_logits, hidden_state):
        """
        expand a node using the value, reward and policy prediction
        obtained from the neural network.
        """
        self.to_play=to_play
        self.reward=reward
        self.hidden_state=hidden_state

        policy_values=tf.nn.softmax(policy_logits).numpy()
        policy={a: policy_values[i] for i, a in enumerate(actions)}
        for action, prob in policy.items():
            self.children[action]=Node(prob)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions=list(self.children.keys())
        noise=numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac=exploration_fraction

        for a, n in zip(actions, noise):
            self.children[a].prior=self.children[a].prior *
                (1 - frac) + n * frac


class MinMaxStats:
    def __init__(self):
        self.min=-float("inf")
        self.max=float("inf")

    def update(self, value):
        self.min=min(self.min, value)
        self.max=max(self.max, value)

    def normalize(self, value):
        if self.max > self.min:
            return (value - self.min)/(self.max - self.min)
        else:
            return value


class MCTS:
    def __init__(self, config):
        self.config=config
        self.num_players=len(config.players)

    def run(self,
            model,
            observation,
            legal_actions,
            to_play,
            add_exploration_noise,
            override_root_with = False):
        if override_root_with:
            root=override_root_with
            root_predicted_value=None
        else:
            root=Node(0)

            (root_predicted_value, reward, policy_logits,
             hidden_state)=model.initial_inference()

            root_predicted_value=models.support_to_scalar(
                root_predicted_value, self.config.support_size)
            reward=models.support_to_scalar(
                reward, self.config.support_size)

            assert (
                legal_actions
            ), f"Legal actions should not be an empty array. Got {legal_actions}."
            assert set(legal_actions).issubset(
                set(self.config.action_space)
            ), "Legal actions should be a subset of the action space."

            root.expand(legal_actions, to_play, reward,
                        policy_logits, hidden_state)

        if add_exploration_noise:
            root.add_exploration_noise(
                self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

        min_max_stats = MinMaxStats()

        max_tree_depth = 0
        for _ in range(self.config.num_simulations):
            virtual_to_play = to_play
            node = root
            search_path = [node]
            current_tree_depth = 0

            while node.expanded():
                current_tree_depth += 1
                action, node = self.select_child(node, min_max_stats)
                search_path.append(node)

                if virtual_to_play + 1 < self.num_players:
                    virtual_to_play = self.config.players[virtual_to_play + 1]
                else:
                    virtual_to_play = self.config.players[0]

            parent = search_path[-2]
            value, reward, policy_logits, hidden_state = model.recurrent_inference(
                parent.hidden_state, action)
            value = model.support_to_scalar(
                value, self.config.support_size)
            reward = model.support_to_scalar(
                reward, self.config.support_size)

            node.expand(self.config.action_space, virtual_to_play,
                        reward, policy_logits, hidden_state)
            self.backpropagate(search_path, value,
                               virtual_to_play, min_max_stats)

        extra_info = {
            "max_tree_depth": max_tree_depth,
            "root_predicted_value": root_predicted_value,
        }
        return root, extra_info

    def select_child(self, node, min_max_stats):
        """
        Select the child with the highest UCB score.
        """
        max_ucb = max(self.ucb_score(node, child, min_max_stats)
                      for _, child in node.children.items())

        action = np.random.choice([
            a for a, child in node.children.items
            if self.ucb_score(node, child, min_max_stats) == max_ucb
        ])

        return action, node.children[action]

    def ucb_score(self, parent, child, min_max_stats):
        pb_c = math.log(
            (parent.visit_count + self.config.pb_c_base + 1)/self.config.pb_c_base)
        pb_c += self.config.pb_c_init
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

        prior_score = pb_c * child.prior

        value_score = 0
        if child.visit_count > 0:
            # account for self-play
            value_score = child.reward + self.config.discount * \
                (child.value() if self.num_players == 1 else -child.value())

            value_score = min_max_stats.normalize(value_score)

        return prior_score + value_score

    def backpropagate(self, search_path, value, to_play, min_max_stats):
        """
        At the end of a simulation, we propagate the evaluation all the way up the tree
        to the root.
        """
        if self.num_players == 1:
            for node in reversed(search_path):
                node.value_sum += value
                node.visit_count += 1
                new_value = node.reward + self.config.discount*node.value()
                min_max_stats.update(new_value)
                value = new_value
        elif self.num_players == 2:
            for node in reversed(search_path):
                if node.to_play == virtual_to_play:
                    node.value_sum += value
                else:
                    node.value_sum -= value
                node.visit_count += 1

                min_max_stats.update(
                    node.reward + self.config.discount * -node.value())

                value = (
                    -node.reward if node.to_play == to_play else node.reward
                ) + self.config.discount * value
        else:
            raise NotImplementedError(
                "More than two player mode not implemented.")
