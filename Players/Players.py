
from Games.GameLogic import BaseGame
from ExIt.Expert.Mcts import Mcts
from ExIt.Policy import e_greedy_action
from ExIt.Apprentice.RandomPredictor import RandomPredictor
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

    def move(self, state: BaseGame, randomness=False, verbose=False):
        return self.move_random(state), None, None

    def new_player(self):
        return RandomPlayer()


class StaticMinimaxPlayer(BasePlayer):
    """ Static Minimax Player with a fixed depth """

    def __init__(self, depth=2):
        super().__init__()
        self.depth = depth
        self.minimax = Minimax(fixed_depth=depth)
        self.predictor = RandomPredictor()
        self.__name__ = type(self).__name__ + "_depth-" + str(depth)

    def move(self, state: BaseGame, randomness=False, verbose=False):
        a, best_a, v = self.minimax.search(
            state=state,
            predictor=self.predictor,
            search_time=None,
            always_exploit=True
        )
        if randomness:
            state.advance(a)
            return a
        state.advance(best_a)
        return best_a

    def new_player(self):
        return StaticMinimaxPlayer(depth=self.depth)


class NnMctsPlayer(BaseExItPlayer):
    """ Player that uses MCTS as the expert and NN as the apprentice """

    def __init__(self, c=sqrt(2), policy=Policy.OFF, use_custom_loss=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(use_custom_loss=use_custom_loss),
                expert=Mcts(c=c),
                policy=policy
            )
        )
        self.c = c
        self.policy = policy
        self.use_custom_loss = use_custom_loss

    def new_player(self):
        return NnMctsPlayer(c=self.c, policy=self.policy, use_custom_loss=self.use_custom_loss)


class NnMinimaxPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None, policy=Policy.OFF, use_custom_loss=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(use_custom_loss=use_custom_loss),
                expert=Minimax(fixed_depth=fixed_depth),
                policy=policy
            )
        )
        self.fixed_depth = fixed_depth
        self.policy = policy
        self.use_custom_loss = use_custom_loss

    def new_player(self):
        return NnMinimaxPlayer(fixed_depth=self.fixed_depth, policy=self.policy, use_custom_loss=self.use_custom_loss)


class NnAlphaBetaPlayer(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None, policy=Policy.OFF, use_custom_loss=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(use_custom_loss=use_custom_loss),
                expert=Minimax(fixed_depth=fixed_depth, use_alpha_beta=True),
                policy=policy
            )
        )
        self.fixed_depth = fixed_depth
        self.policy = policy
        self.use_custom_loss = use_custom_loss

    def new_player(self):
        return NnAlphaBetaPlayer(fixed_depth=self.fixed_depth, policy=self.policy, use_custom_loss=self.use_custom_loss)


class NnAbBranch(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, fixed_depth=None, branch_prob=0.25, use_custom_loss=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(use_custom_loss=use_custom_loss),
                expert=Minimax(fixed_depth=fixed_depth, use_alpha_beta=True),
                policy=Policy.OFF,
                always_exploit=True,
                memory=MemorySet(),
                branch_prob=branch_prob
            )
        )
        self.fixed_depth = fixed_depth
        self.branch_prob = branch_prob
        self.use_custom_loss = use_custom_loss

    def new_player(self):
        return NnAbBranch(fixed_depth=self.fixed_depth, branch_prob=self.branch_prob, use_custom_loss=self.use_custom_loss)


class NnMctsBranch(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self, c=sqrt(2), branch_prob=0.25, use_custom_loss=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(use_custom_loss=use_custom_loss),
                expert=Mcts(c=c),
                policy=Policy.OFF,
                always_exploit=True,
                memory=MemorySet(),
                branch_prob=branch_prob
            )
        )
        self.c = c
        self.branch_prob = branch_prob
        self.use_custom_loss = use_custom_loss

    def new_player(self):
        return NnMctsBranch(c=self.c, branch_prob=self.branch_prob, use_custom_loss=self.use_custom_loss)


class NnMinimaxBranch(BaseExItPlayer):
    """ Player that uses Minimax as expert and NN as apprentice """

    def __init__(self):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(switch=True),
                policy=Policy.OFF
            )
        )

    def new_player(self):
        return NnMinimaxBranch()

