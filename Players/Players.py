
from Games.BaseGame import BaseGame
from ExIt.Expert.Mcts import Mcts
from ExIt.Apprentice.Nn import Nn
from ExIt.Expert.MiniMax import MiniMax
from ExIt.ExpertIteration import ExpertIteration
from random import choice as rnd_choice
from Players.BasePlayer import BasePlayer, BaseExItPlayer
from math import sqrt


class RandomPlayer(BasePlayer):
    """ Player that plays random moves """

    def move(self, game: BaseGame):
        legal_moves = game.get_possible_actions()
        action_index = rnd_choice(legal_moves)
        game.advance(action_index=action_index)


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

    def __init__(self):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=MiniMax()
            )
        )
