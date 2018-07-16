
from Games.GameLogic import BaseGame
from ExIt.Expert.Mcts import Mcts
from ExIt.Apprentice.Nn import Nn
from ExIt.Expert.Minimax import Minimax
from ExIt.Expert.Mcts import Mcts
from ExIt.ExpertIteration import ExpertIteration
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from ExIt.Memory import MemoryList, MemorySet
from ExIt.Policy import Policy
from math import sqrt


class RandomPlayer(BasePlayer):
    """ Player that plays random moves """

    def __init__(self):
        super().__init__()
        self.__name__ = type(self).__name__

    def move(self, state: BaseGame, randomness=False):
        return self.move_random(state), None, None

    def new_player(self):
        return RandomPlayer()


class NnMctsPlayer(BaseExItPlayer):
    """ Player that uses MCTS as the expert and NN as the apprentice """

    def __init__(self, c=sqrt(2), policy=Policy.OFF):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Mcts(c=c),
                policy=policy
            )
        )
        self.c = c
        self.policy = policy

    def new_player(self):
        return NnMctsPlayer(c=self.c, policy=self.policy)


class NnMinimaxPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None, policy=Policy.OFF):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth),
                policy=policy
            )
        )
        self.fixed_depth = fixed_depth
        self.policy = policy

    def new_player(self):
        return NnMinimaxPlayer(fixed_depth=self.fixed_depth, policy=self.policy)


class NnAlphaBetaPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None, policy=Policy.OFF):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth, use_alpha_beta=True),
                policy=policy
            )
        )
        self.fixed_depth = fixed_depth
        self.policy = policy

    def new_player(self):
        return NnAlphaBetaPlayer(fixed_depth=self.fixed_depth, policy=self.policy)


class LarsPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None, branch_prob=0.25):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth, use_alpha_beta=True),
                policy=Policy.OFF,
                use_exploration_policy=False,
                memory=MemorySet(),
                branch_prob=branch_prob
            )
        )
        self.branch_prob = branch_prob

    def new_player(self):
        return LarsPlayer(branch_prob=self.branch_prob)
