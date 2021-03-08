from abc import ABC, abstractmethod
import numpy as np
import tensorflow as tf


class AbstractNetwork(ABC, tf.keras.Model):
    def __init__():
        super(AbstractNetwork, self).__init__()
        pass

    @abstractmethod
    def initial_inference(self, observation):
        pass

    @abstractmethod
    def recurrent_inference(self, encoded_state, action):
        pass


class ResidualBlock(tf.keras.layers.Layer):
    """
    residual block used in alphago zero, alpha zero, and muzero paper
    """

    def __init__(self, num_channels=128):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, strides=1, padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, strides=1, padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()

    def call(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class RepresentationNetwork(tf.keras.Model):
    def __init__(self):
        super(RepresentationNetwork, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            128, kernel_size=3, strides=2, padding="same", activation="relu")

        self.res_blocks1 = [ResidualBlock(128) for _ in range(2)]

        self.conv2 = tf.keras.layers.Conv2D(
            256, kernel_size=3, strides=2, padding="same", activation="relu")

        self.res_blocks2 = [ResidualBlock(256) for _ in range(3)]

        self.avg_pool1 = tf.keras.layers.AveragePooling2D(strides=2)

        self.res_blocks3 = [ResidualBlock(256) for _ in range(3)]

        self.avg_pool2 = tf.keras.layers.AveragePooling2D(strides=2)

    def call(self, x):
        x = self.conv1(x)
        for block in self.res_blocks1:
            x = block(x)
        x = self.conv2(x)
        for block in self.res_blocks2:
            x = block(x)
        x = self.avg_pool1(x)
        for block in self.res_blocks3:
            x = block(x)
        x = self.avg_pool2(x)
        return x


class DynamicsNetwork(tf.keras.Model):
    def __init__(self,
                 num_channels,
                 full_support_size
                 ):
        super(DynamicsNetwork, self).__init__()

        self.conv = tf.keras.layers.Conv2D(
            num_channels, kernel_size=3, strides=1, padding="same")
        self.bn = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.res_blocks = [ResidualBlock(num_channels) for i in range(3)]

        self.conv_reward = tf.keras.layers.Conv2D(1, kernel_size=1)
        self.bn_reward = tf.keras.layers.BatchNormalization()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(full_support_size, activation="tanh")

    def build(self, input_shape):
        self.batch_size = input_shape.as_list()[0]

    def call(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        for block in self.res_blocks:
            x = block(x)
        state = x
        x = self.conv_reward(x)
        x = self.bn_reward(x)
        x = tf.reshape(x, [-1, 1])
        x = tf.reshape(x, [self.batch_size, -1])
        x = self.fc1(x)
        reward = self.fc2(x)
        return state, reward


class PredictionNetwork(tf.keras.Model):
    def __init__(self,
                 action_space_size,
                 full_support_size
                 ):
        super(PredictionNetwork, self).__init__()
        self.action_space_size = action_space_size

        self.conv_policy = tf.keras.layers.Conv2D(2, kernel_size=1)
        self.bn_policy = tf.keras.layers.BatchNormalization()
        self.fc_policy = tf.keras.layers.Dense(
            self.action_space_size, activation="relu")

        self.conv_value = tf.keras.layers.Conv2D(1, kernel_size=1)
        self.bn_value = tf.keras.layers.BatchNormalization()
        self.relu_value = tf.keras.layers.ReLU()
        self.fc_value1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc_value2 = tf.keras.layers.Dense(
            full_support_size, activation="tanh")

    def build(self, input_shape):
        self.batch_size = input_shape.as_list()[0]

    def call(self, x):
        policy = self.conv_policy(x)
        policy = self.bn_policy(policy)
        policy = self.fc_policy(policy)

        value = self.conv_value(x)
        value = self.bn_value(value)
        value = tf.reshape(value, [self.batch_size, -1])
        value = self.fc_value1(value)
        value = self.fc_value2(value)
        return policy, value


class MuZeroResidualNetwork(AbstractNetwork):
    def __init__(self,
                 observation_shape,
                 action_space_size,
                 num_channels,
                 support_size):
        super(MuZeroResidualNetwork, self).__init__()
        self.observation_shape = observation_shape
        self.action_space_size = action_space_size
        self.num_channels = num_channels
        self.support_size = support_size
        self.full_support_size = 2 * support_size + 1

        self.representation_network = RepresentationNetwork()
        self.dynamics_network = DynamicsNetwork(
            self.num_channels, self.full_support_size)
        self.prediction_network = PredictionNetwork(
            self.action_space_size, self.full_support_size)

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        return normalize_encoded_state(encoded_state)

    def dynamics(self, encoded_state, action):
        state_shape = encoded_state.shape.as_list()
        action_shape = state_shape[:-1] + [self.action_space_size]
        ones = tf.ones(action_shape)
        action_one_hot = tf.one_hot(action, depth=self.action_space_size)
        action_one_hot = action_one_hot[:, tf.newaxis, tf.newaxis, :]
        action_one_hot = ones*action_one_hot

        x = tf.cat([encoded_state, action_one_hot), axis = -1)
        next_encoded_state, reward = self.dynamics_network(x)

        return normalize_encoded_state(next_encoded_state), reward

    def initial_inference(self, observation):
        encoded_state = self.representation(observation)
        policy_logits, value = self.prediction(encoded_state)
        reward = tf.zeros(observation.shape.as_list()[0])
        return (value, reward, policy_logits, encoded_state)

    def recurrent_inference(self, encoded_state, action):
        next_encoded_state, reward = self.dynamics(encoded_state, action)
        policy_logits, value = self.prediction(next_encoded_state)
        return value, reward, policy_logits, next_encoded_state


def normalize_encoded_state(encoded_state):
    shape = encoded_state.shape.as_list()
    batch_size = shape[0]
    reshaped_encoded_state = tf.reshape(encoded_state, [batch_size, -1])
    min_encoded_state = tf.math.reduce_min(reshaped_encoded_state, axis=-1)
    max_encoded_state = tf.math.reduce_max(reshaped_encoded_state, axis=-1)
    scaled_encoded_state = reshaped_encoded_state - min_encoded_state
    normalized_encoded_state = scaled_encoded_state / \
        (max_encoded_state - min_encoded_state + 1e-5)

    return tf.reshape(normalized_encoded_state, shape)


def support_to_scalar(logits, support_size):
    probabilities = tf.nn.softmax(logits, axis=-1)
    support = tf.expand_dims(
        tf.range(-support_size, support_size + 1, dtype=tf.float32), axis=1)
    print(probabilities)
    print(support)
    x = tf.linalg.matmul(probabilities, support)
    x = tf.math.sign(x) * (tf.math.square(
        (tf.math.sqrt(1 + 4 * 0.001 * (tf.math.abs(x) + 1 + 0.001)) - 1) / (2 * 0.001))
        - 1.0
    )

    return x


def scalar_to_support(x, support_size):
    x = tf.math.sign(x)*(tf.math.sqrt(tf.math.abs(x) + 1.0) - 1.0) + 0.001*x
    x = tf.clip_by_value(x, -support_size, support_size - 1e-6)
    floor = tf.math.floor(x)
    prob = x - floor

    index_low = floor + support_size
    flat_index_low = tf.reshape(tf.cast(index_low, dtype=tf.int32), [-1])

    support_axis_dim = 2 * support_size + 1
    flat_index_low += tf.range(tf.size(x))*support_axis_dim
    flat_index_high = flat_index_low + 1

    flat_prob = tf.reshape(prob, [-1])

    flat_logits = tf.zeros(tf.size(x)*support_axis_dim)
    flat_logits = tf.tensor_scatter_nd_update(flat_logits, tf.reshape(
        flat_index_low, [-1, 1]), 1.0 - flat_prob)
    flat_logits = tf.tensor_scatter_nd_update(flat_logits, tf.reshape(
        flat_index_high, [-1, 1]), flat_prob)

    return tf.reshape(flat_logits, x.shape.as_list() + [support_axis_dim])


if __name__ == "__main__":
    rep_net = RepresentationNetwork()
    img_input = tf.random.uniform([4, 96, 96, 128])
    output = rep_net(img_input)
    rep_net.summary()
    print(output.shape)

    dyn_net = DynamicsNetwork(256, 601)
    dyn_input = tf.random.uniform([4, 6, 6, 257])
    output = dyn_net(dyn_input)
    dyn_net.summary()
    print(output[0].shape)
    print(output[1].shape)

    pred_net = DynamicsNetwork(256, 601)
    pred_input = tf.random.uniform([4, 6, 6, 256])
    output = pred_net(pred_input)
    pred_net.summary()
    print(output[0].shape)
    print(output[1].shape)

    logits = np.array([[-1.0, 0.1, 1.4, 2.5, 2.0]]).astype(np.float32)
    print(support_to_scalar(logits, support_size=2))

    print(scalar_to_support(tf.constant(
        [[-1.4, 1.3], [1.0, -1.9]]), support_size=2))
