
from Games.GameLogic import BaseGame
from ExIt.Apprentice.RandomPredictor import RandomPredictor
from ExIt.Apprentice.Nn import Nn
from ExIt.Expert.Minimax import Minimax
from ExIt.Expert.Mcts import Mcts
from ExIt.ExpertIteration import ExpertIteration
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from ExIt.Policy import Policy


GROWING_SEARCH_VALUE = 0.0001


def get_grow_search_val(growing_search):
    return GROWING_SEARCH_VALUE if growing_search else None


""" ****************************************
        Random
    **************************************** """


class RandomPlayer(BasePlayer):
    """ Player that plays random moves """

    def __init__(self):
        super().__init__()
        self.__name__ = type(self).__name__

    def move(self, state: BaseGame, randomness=False, verbose=False):
        return self.move_random(state), None, None

    def new_player(self):
        return RandomPlayer()


""" ****************************************
        Static players
    **************************************** """


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


""" ****************************************
        MCTS
    **************************************** """


class NnMctsPlayer(BaseExItPlayer):

    def __init__(self, policy=Policy.OFF, growing_search=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Mcts(),
                policy=policy,
                growing_search=get_grow_search_val(growing_search),
            )
        )
        self.policy = policy
        self.growing_search = growing_search
        if growing_search:
            self.set_search_time(0.0)

    def new_player(self):
        return NnMctsPlayer(policy=self.policy, growing_search=self.growing_search)


""" ****************************************
        Minimax and Alpha Beta
    **************************************** """


class NnMinimaxPlayer(BaseExItPlayer):

    def __init__(self, use_ab=False, fixed_depth=None, policy=Policy.OFF,
                 growing_search=False, soft_z=False, growing_depth=True):
        if fixed_depth is not None and growing_search:
            raise Exception("Cannot have a fixed search depth and growing search timer!")
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth, use_ab=use_ab),
                policy=policy,
                growing_search=get_grow_search_val(growing_search),
                soft_z=soft_z,
                growing_depth=growing_depth
            )
        )
        self.use_ab = use_ab
        self.fixed_depth = fixed_depth
        self.policy = policy
        self.growing_search = growing_search
        self.soft_z = soft_z
        self.growing_depth = growing_depth
        if growing_search:
            self.set_search_time(0.0)

    def new_player(self):
        return NnMinimaxPlayer(
            use_ab=self.use_ab, fixed_depth=self.fixed_depth, policy=self.policy,
            growing_search=self.growing_search, soft_z=self.soft_z, growing_depth=self.growing_depth
        )
