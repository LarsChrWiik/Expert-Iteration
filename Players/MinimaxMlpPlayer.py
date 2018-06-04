
from Players.Base import BaseExItPlayer
from ExIt.Apprentice.MLP import MLP
from ExIt.Expert.MiniMax import MiniMax
from ExIt.ExpertIteration import ExpertIteration


class MinimaxMlpPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, minimax_depth):
        super().__init__(ex_it_algorithm=ExpertIteration(
            apprentice=MLP(), expert=MiniMax(depth=minimax_depth)))
