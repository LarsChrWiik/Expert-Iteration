
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame


class BaseExpert:
    """ Class for policy improvement logic """

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        """ Do policy improvement for a given state.
        :return: action_index """
        raise NotImplementedError("Please Implement this method")

