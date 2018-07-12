
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame


class BaseExpert:
    """ Class for policy improvement logic """

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time, use_off_policy):
        """ Do policy improvement for a given state.
        :return: action_index """
        raise NotImplementedError("Please Implement this method")
