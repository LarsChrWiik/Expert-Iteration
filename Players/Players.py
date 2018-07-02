
from Games.GameLogic import BaseGame
from ExIt.Expert.Mcts import Mcts
from ExIt.Apprentice.Nn import Nn
from ExIt.Expert.Minimax import Minimax
from ExIt.Expert.Minimax import Minimax
from ExIt.ExpertIteration import ExpertIteration
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from math import sqrt


class RandomPlayer(BasePlayer):
    """ Player that plays random moves """

    def move(self, game: BaseGame, randomness=False):
        self.move_random(game=game)


class NnMctsPlayer(BaseExItPlayer):
    """ Player that uses MCTS as the expert and NN as the apprentice """

    def __init__(self, c=sqrt(2)):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Mcts(c=c)
            )
        )


class NnMinimaxPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth)
            )
        )


class NnAlphaBetaPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth, use_alpha_beta=True)
            )
        )
