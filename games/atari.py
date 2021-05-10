import os
import time

import cv2
import numpy as np
from ale_py import ALEInterface

from games.abstract_game import AbstractGame


class AtariGame(AbstractGame):
    def __init__(self, game_name="space_invaders", seed=None, img_size=(96, 96)):
        super().__init__()
        self.ale = ALEInterface()
        if seed is None:
            seed = 123
        self.ale.setInt("random_seed", seed)
        if game_name is None:
            self.game_name = "space_invaders"
        else:
            self.game_name = game_name
        self.rom_file = f"/home/alanxu/Documents/{game_name}.bin"
        self.ale.loadROM(self.rom_file)
        self.img_size = img_size

    def legal_actions(self):
        return self.ale.getLegalActionSet()

    def reset(self):
        self.ale.reset_game()
        observation = cv2.resize(self.ale.getScreenRGB(), self.img_size)
        observation = np.array(observation, dtype="float32") / 255.0

        return observation

    def step(self, action):
        reward = self.ale.act(action)
        new_observation = cv2.resize(self.ale.getScreenRGB(), self.img_size)
        new_observation = np.array(new_observation, dtype="float32") / 255.0
        game_over = self.ale.game_over()

        return new_observation, reward, game_over

    def render(self):
        return cv2.imshow("atari", self.ale.getScreenRGB())

    def close(self):
        return


if __name__ == "__main__":
    g = AtariGame(seed=123)
    img = g.reset()
    print(img.shape)

    t0 = time.time()
    count = 0
    for i in range(1000):
        a = np.random.randint(18)
        _, _, done = g.step(a)
        count += 1
        if done:
            break
    print("steps: {}".format(count))
    print(time.time() - t0)

    # img = cv2.resize(img, (96, 96))

    cv2.imshow("img", img)

    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
