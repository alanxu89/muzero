import copy
import os
import numpy as np

import tensorflow as tf


class SharedStorage:
    def __init__(self, config, checkpoint):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)

    def save_checkpoint(self, path=None):
        if path is None:
            path = os.path.join(self.config.results_path, "model_checkpoints")
        self.config.model.save_weights(path)

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)


if __name__ == "__main__":
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, input_shape=(16,)),
        tf.keras.layers.Dense(16),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    model.summary()
    test_input = np.random.random((128, 16))
    test_target = np.random.random((128, 1))
    model.fit(test_input, test_target)

    # model.save_weights("my_model")

    weights = model.get_weights()
    print("================")
    for w in weights:
        print(w.shape)
