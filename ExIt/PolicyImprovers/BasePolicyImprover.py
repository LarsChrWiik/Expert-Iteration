
from ExIt.PolicyValuePredictors import BasePolicyValuePredictor
from Games.BaseGame import BaseGame


class BasePolicyImprover:
    """
    Class for policy improvement logic.
    """

    @staticmethod
    def search_and_store(game: BaseGame, predictor: BasePolicyValuePredictor):
        """
        Do policy improvement for a given state.

        :param game: BaseGame.
        :param predictor: BasePolicyValuePredictor.
        :return: action_index
        """
        # TODO: Need a stopping criteria: Time?

        # TODO: When stopping criteria is met
        # TODO: then return an action index.
        # TODO: Might want to have some randomness???
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def search(self, game: BaseGame, predictor: BasePolicyValuePredictor):
        # Same as "search_and_store" but not store.
        raise NotImplementedError("Please Implement this method")
