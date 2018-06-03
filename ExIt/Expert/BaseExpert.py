
from ExIt.Apprentice import BaseApprentice
from Games.BaseGame import BaseGame


class BaseExpert:
    """
    Class for policy improvement logic.
    """

    def start(self, state: BaseGame, predictor: BaseApprentice):
        """
        Do policy improvement for a given state.

        :return: action_index
        """
        # TODO: Might want to have some randomness???
        raise NotImplementedError("Please Implement this method")

