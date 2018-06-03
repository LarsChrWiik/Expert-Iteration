
from Players.BasePlayers.BaseExItPlayer import BaseExItPlayer
from ExIt.Apprentice.MLP import MLP
from ExIt.Expert.MCTS import MCTS
from ExIt.ExpertIteration import ExpertIteration


class NnMctsPlayer(BaseExItPlayer):
    """
    Player that uses:
        - MCTS as Policy improvement operator.
        - NN as Policy and evaluation predictor.
    """

    def __init__(self):
        super().__init__(exIt_algorithm=ExpertIteration(
            policy_value_predictor=MLP(),
            policy_improvement_operator=MCTS()
        ))
