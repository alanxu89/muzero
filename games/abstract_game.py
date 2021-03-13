from abc import ABC, abstractmethod


class AbstractGame(ABC):

    @abstractmethod
    def __init__(self, seed=None):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def to_play(self):
        pass

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
