import os
import cv2

from ale_py import ALEInterface

from games.abstract_game import AbstractGame


class AtariGame(AbstractGame):
    def __init__(self, rom_file=None, seed=None, img_size=(96, 96)):
        super().__init__()
        self.ale = ALEInterface()
        if seed is None:
            seed = 123
        self.ale.setInt("random_seed", seed)
        if rom_file is None:
            rom_file = "/Users/alanquantum/MachineLearning/atari_roms/space_invaders.bin"
        self.ale.loadROM(rom_file)
        self.img_size = img_size

    def legal_actions(self):
        return self.ale.getLegalActionSet()

    def reset(self):
        self.ale.reset_game()

        return cv2.resize(self.ale.getScreenRGB(), self.img_size)

    def step(self, action):
        reward = self.ale.act(action)
        new_observation = cv2.resize(self.ale.getScreenRGB(), self.img_size)
        game_over = self.ale.game_over()

        return new_observation, reward, game_over

    def render(self):
        # if cv2 is available, use cv2.imshow(self.ale.getScreenRGB())
        return cv2.imshow("atari", self.ale.getScreenRGB())

    def close(self):
        return


if __name__ == "__main__":
    g = AtariGame(
        "/Users/alanquantum/MachineLearning/atari_roms/space_invaders.bin", 123)
    # screen_rgb = g.render()
    img = g.reset()
    print(img.shape)

    # img = cv2.resize(img, (96, 96))

    cv2.imshow("img", img)

    cv2.waitKey(0)

    # closing all open windows
    cv2.destroyAllWindows()
