
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame


class BaseExpert:
    """ Class for policy improvement logic """

    def __init__(self):
        self.__name__ = type(self).__name__

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time, use_exploration_policy):
        """ Do policy improvement for a given state.
        :return: action_index """
        raise NotImplementedError("Please Implement this method")
