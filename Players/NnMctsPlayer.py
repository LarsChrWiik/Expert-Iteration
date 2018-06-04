
from Players.Base import BaseExItPlayer
from ExIt.Apprentice.MLP import MLP
from ExIt.Expert.MCTS import MCTS
from ExIt.ExpertIteration import ExpertIteration


class NnMctsPlayer(BaseExItPlayer):
    """Player that uses MCTS as the expert and NN as the apprentice """

    def __init__(self):
        super().__init__(ex_it_algorithm=ExpertIteration(apprentice=MLP(), expert=MCTS()))
