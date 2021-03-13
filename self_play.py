import numpy as np

import models


class Node:
    def __init__(self, prior):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

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
        self.to_play = to_play
        self.reward = reward
        self.hidden_state = hidden_state

        policy_values = tf.nn.softmax(policy_logits).numpy()
        policy = {a: policy_values[i] for i, a in enumerate(actions)}
        for action, prob in policy.items():
            self.children[action] = Node(prob)

    def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
        """
        At the start of each search, we add dirichlet noise to the prior of the root to
        encourage the search to explore new actions.
        """
        actions = list(self.children.keys())
        noise = numpy.random.dirichlet([dirichlet_alpha] * len(actions))
        frac = exploration_fraction

        for a, n in zip(actions, noise):
            self.children[a].prior = self.children[a].prior * \
                (1 - frac) + n * frac


class MinMaxStats:
    def __init__(self):
        self.min = -float("inf")
        self.max = float("inf")

    def update(self, value):
        self.min = min(self.min, value)
        self.max = max(self.max, value)

    def normalize(self, value):
        if self.max > self.min:
            return (value - self.min)/(self.max - self.min)
        else:
            return value


class MCTS:
    def __init__(self, config):
        self.config = config
        self.num_players = len(config.players)

    def run(self,
            model,
            observation,
            legal_actions,
            to_play,
            add_exploration_noise,
            override_root_with=False):
        if override_root_with:
            root = override_root_with
            root_predicted_value = None
        else:
            root = Node(0)

            (root_predicted_value, reward, policy_logits,
             hidden_state) = model.initial_inference()

            root_predicted_value = models.support_to_scalar(
                root_predicted_value, self.config.support_size)
            reward = models.support_to_scalar(
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
