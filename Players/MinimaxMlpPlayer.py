
from Players.BasePlayers.BaseExItPlayer import BaseExItPlayer
from ExIt.Apprentice.MLP import MLP
from ExIt.Expert.MiniMax import MiniMax
from ExIt.ExpertIteration import ExpertIteration


class MinimaxMlpPlayer(BaseExItPlayer):
    """
    Player that uses:
        - Minimax as Policy improvement operator.
        - NN as Policy and evaluation predictor.
    """

    def __init__(self, minimax_depth):
        super().__init__(
            exIt_algorithm=ExpertIteration(
                apprentice=MLP(),
                expert=MiniMax(depth=minimax_depth)
            )
        )
