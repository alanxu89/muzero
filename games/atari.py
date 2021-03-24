import os
import datetime

from ale_py import ALEInterface

from games.abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        self.seed = 0
        self.max_num_gpus = None

        self.observation_shape = (96, 96, 3)
        self.action_space = list(range(4))
        self.players = list(range(1))
        self.stacked_observations = 32

        self.muzero_player = 0
        self.opponent = None

        self.num_workers = 6
        self.selfplay_on_gpu = False
        self.max_moves = 27000
        self.num_simulations = 50
        self.discount = 0.997
        self.temperature_threshold = None

        self.root_dirichlet_alpha = 0.25
        self.root_exploration_fraction = 0.25

        # UCB formula
        self.pb_c_base = 19652
        self.pb_c_init = 1.25

        # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))
        self.support_size = 300

        # Residual Network
        self.blocks = 16  # Number of blocks in the ResNet
        self.channels = 256  # Number of channels in the ResNet
        self.reduced_channels_reward = 256  # Number of channels in reward head
        self.reduced_channels_value = 256  # Number of channels in value head
        self.reduced_channels_policy = 256  # Number of channels in policy head
        # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_reward_layers = [256, 256]
        # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_value_layers = [256, 256]
        # Define the hidden layers in the policy head of the prediction network
        self.resnet_fc_policy_layers = [256, 256]

        # Training
        #  Path to store the model weights and TensorBoard logs
        self.results_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                         "../results",
                                         os.path.basename(__file__)[:-3],
                                         datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
                                         )
        self.save_model = True
        self.training_steps = int(1e6)
        self.batch_size = 1024
        # Number of training steps before using the model for self-playing
        self.checkpoint_interval = int(1e3)
        # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
        self.value_loss_weight = 0.25
        self.train_on_gpu = True

        self.optimizer = "SGD"
        self.weight_decay = 1e-4
        self.momentum = 0.9

        self.lr_init = 0.05
        self.lr_decay_rate = 0.1
        self.lr_decay_steps = int(350e3)

        # Replay Buffer
        self.replay_buffer_size = int(1e6)
        self.num_unroll_steps = 5
        # Number of steps in the future to take into account for calculating the target value
        self.td_steps = 10
        # Prioritized Replay (See paper appendix Training), select in priority the elements
        # in the replay buffer which are unexpected for the network
        self.PER = True
        # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
        self.PER_alpha = 1

        # Reanalyze (See paper appendix Reanalyse)
        # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
        self.use_last_model_value = True
        self.reanalyse_on_gpu = False

        # Adjust the self play / training ratio to avoid over/underfitting
        self.self_play_delay = 0  # Number of seconds to wait after each played game
        self.training_delay = 0  # Number of seconds to wait after each training step
        # Desired training steps per self played step ratio. Equivalent to a synchronous version,
        # training can take much longer. Set it to None to disable it
        self.ratio = None


class AtariGame(AbstractGame):
    def __init__(self, rom_file, seed):
        super().__init__()
        self.ale = ALEInterface()
        self.ale.setInt("random_seed", seed)
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
