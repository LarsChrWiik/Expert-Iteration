
from ExIt.PolicyValuePredictors import BasePolicyValuePredictor
from ExIt.PolicyImprovers import BasePolicyImprover
from Games.BaseGame import BaseGame


class ExItAlgorithm:
    """
    Abstract class for Expert Iteration algorithms.
    """

    # NN.
    policy_value_predictor = None

    # MCTS or AB-search.
    policy_improvement_operator = None

    def __init__(
            self,
            policy_value_predictor: BasePolicyValuePredictor,
            policy_improvement_operator: BasePolicyImprover
    ):
        self.policy_value_predictor = policy_value_predictor
        self.policy_improvement_operator = policy_improvement_operator

    def calculate_best_action(self, game: BaseGame):
        """
        This is used after the Expert Iteration has completed.

        :param game: BaseGame object.
        :return: int, action index.
        """
        pass

    def start_expert_iteration(self, game_original: BaseGame):
        """
        Start Expert Iteration to master the given game.
        This process might take some time.
        """

        # TODO: Add another for loop for several games.
        game = game_original.get_state_copy()

        """
        Play a game against it self. 
        Store:
            - Moves. 
            - Policy improver for each move. 
            - Reward at end. 
        """
        # Test with one iteration first.
        while not game.has_finished():
            self.exIt_base_state(game)

        # Retrain the Policy and evaluation predictor.
        # TODO: Retrain.

    def exIt_base_state(self, game: BaseGame):
        action_index = self.policy_improvement_operator.search_and_store(
            game=game,
            predictor=self.policy_value_predictor
        )
        game.advance(action_index=action_index)
