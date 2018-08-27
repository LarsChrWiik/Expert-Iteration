
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame


class BaseExpert:
    """ Class for the tree search algorithm used for policy improvement """

    def __init__(self):
        self.__name__ = type(self).__name__

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time, use_exploration_policy):
        """ Do policy improvement for a given state.
        :return: a_explore, a_optimal, soft-z """
        raise NotImplementedError("Please Implement this method")
