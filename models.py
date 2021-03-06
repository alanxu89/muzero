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


class MuZeroFullyConnectedNetwork(AbstractNetwork):
    def __init__(self,
                 observation_shape,
                 stacked_observations,
                 action_space_size,
                 encoding_size,
                 fc_reward_layers,
                 fc_value_layers,
                 fc_policy_layers,
                 fc_representation_layers,
                 fc_dynamics_layers,
                 support_size
                 ):
        super(MuZeroFullyConnectedNetwork, self).__init__()
        self.x = x

    def initial_inference(self, observation):
        return observation

    def recurrent_inference(self, encoded_state, action):
        return action


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

        #  reduce the original channel by one, since the last dimension is action
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


def mlp(input_size,
        layer_sizes,
        output_size,
        activation="gelu",
        output_activation="gelu"):
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Dense(
        layer_sizes[0], activation=activation, input_shape=(input_size,)))
    for size in layer_sizes[1:]:
        model.add(tf.keras.layers.Dense(size, activation=activation))
    model.add(tf.keras.layers.Dense(
        output_size, activation=output_activation))

    return model


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

    flat_logits = tf.zeros(tf.size(x)*support_axis_dim)
    flat_logits = tf.tensor_scatter_nd_update(flat_logits, tf.reshape(
        flat_index_low, [-1, 1]), tf.reshape(1.0 - prob, [-1]))
    flat_logits = tf.tensor_scatter_nd_update(flat_logits, tf.reshape(
        flat_index_high, [-1, 1]), tf.reshape(prob, [-1]))

    return tf.reshape(flat_logits, x.shape.as_list() + [support_axis_dim])


if __name__ == "__main__":
    # rep_net = RepresentationNetwork()
    # img_input = tf.random.uniform([4, 96, 96, 128])
    # output = rep_net(img_input)
    # rep_net.summary()
    # print(output.shape)

    # dyn_net = DynamicsNetwork(256, 601)
    # dyn_input = tf.random.uniform([4, 6, 6, 257])
    # output = dyn_net(dyn_input)
    # dyn_net.summary()
    # print(output[0].shape)
    # print(output[1].shape)

    # pred_net = DynamicsNetwork(256, 601)
    # pred_input = tf.random.uniform([4, 6, 6, 256])
    # output = pred_net(pred_input)
    # pred_net.summary()
    # print(output[0].shape)
    # print(output[1].shape)

    logits = np.array([[-1.0, 0.1, 1.4, 2.5, 2.0]]).astype(np.float32)
    print(support_to_scalar(logits, support_size=2))

    # print(scalar_to_support(tf.constant(
    #     [[-1.4, 1.3], [1.0, -1.9]]), support_size=2))
