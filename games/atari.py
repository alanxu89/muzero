import os

from ale_py import ALEInterface

from games.abstract_game import AbstractGame


class AtariGame(AbstractGame):
    def __init__(self, rom_file=None, seed=None):
        super().__init__()
        self.ale = ALEInterface()
        if seed is None:
            seed = 123
        self.ale.setInt("random_seed", seed)
        if rom_file is None:
            rom_file = "/Users/alanquantum/MachineLearning/atari_roms/space_invaders.bin"
        self.ale.loadROM(rom_file)

    def legal_actions(self):
        return self.ale.getLegalActionSet()

    def reset(self):
        self.ale.reset_game()

    def step(self, action):
        reward = self.ale.act(action)
        new_observation = self.ale.getScreenRGB()
        game_over = self.ale.game_over()

        return new_observation, reward, game_over

    def render(self):
        # if cv2 is available, use cv2.imshow(self.ale.getScreenRGB())
        return

    def close(self):
        return


if __name__ == "__main__":
    g = AtariGame(
        "/Users/alanquantum/MachineLearning/atari_roms/space_invaders.bin", 123)
    screen_rgb = g.render()
    print(screen_rgb.shape)
