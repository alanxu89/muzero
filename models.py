from abc import ABC, abstractmethod
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
    def __init__(self, num_channels=128):
        super().__init__():
        self.conv1 = tf.keras.layers.Conv2D(128, kernel_size=3, stride=1)
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(128, kernel_size=3, stride=1)
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


class Downsample(tf.keras.Model):
    def __init__(self, in_channels, out_channels, h_w):
        super(DownsampleCNN, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            128, kernel_size=3, stride=2, activation="relu")

        self.res_blocks1 = [ResidualBlock() for _ in range(2)]

        self.conv2 = tf.keras.layers.Conv2D(
            256, kernel_size=3, stride=2, activation="relu")

        self.res_blocks2 = [ResidualBlock() for _ in range(3)]

        self.avg_pool1 = tf.keras.layers.AveragePooling2D(
            kernel_size=3, stride=2)

        self.res_blocks3 = [ResidualBlock() for _ in range(3)]

        self.avg_pool2 = tf.keras.layers.AveragePooling2D(
            kernel_size=3, stride=2)

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


if __name__ == "__main__":
    # net = AbstractNetwork()
    # net.get_weights()
    # net = MuZeroFullyConnectedNetwork(12)
    # print("hello world")
    # print(net.initial_inference([1, 2, 3]))

    model = mlp(10, [8, 8, 8], 12)
    model.summary()
