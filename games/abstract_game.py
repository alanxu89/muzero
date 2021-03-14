from abc import ABC, abstractmethod


class AbstractGame(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def step(self, action):
        """
        Args:
            action : action of the action_space to take.

        Returns:
            The new observation, the reward and a boolean if the game has ended.
        """
        pass

    def to_play(self):
        # return player 0 by default
        return 0

    @abstractmethod
    def legal_actions(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    def close(self):
        pass

    @abstractmethod
    def render(self):
        pass

    def human_to_action(self):
        """
        For multiplayer games, ask the user for a legal action
        and return the corresponding action number.
        """
        choice = input(
            f"Enter the action to play for the player {self.to_play()}: ")
        while int(choice) not in self.legal_actions():
            choice = input("Ilegal action. Enter another action : ")
        return int(choice)

    def expert_agent(self):
        raise NotImplementedError

    def action_to_string(self, action_number):
        return str(action_number)
