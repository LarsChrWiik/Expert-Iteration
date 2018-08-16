
from Games.GameLogic import BaseGame
from ExIt.Apprentice.RandomPredictor import RandomPredictor
from ExIt.Apprentice.Nn import Nn
from ExIt.Expert.Minimax import Minimax
from ExIt.Expert.Mcts import Mcts
from ExIt.ExpertIteration import ExpertIteration
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from ExIt.Policy import Policy, explore
import random
from Matchmaking.GameHandler import GameHandler


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

        _, best_a, v = self.minimax.search(
            state=state,
            predictor=self.predictor,
            search_time=None, # Because of fixed depth.
            always_exploit=True
        )

        lm = state.get_legal_moves()
        a = best_a
        if randomness and len(lm) > 1 and random.uniform(0, 1) < self.rnd_e:
            # Chose non-optimal move.
            a = explore([x for x in lm if x != a])
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

    def __init__(self, policy=Policy.OFF, growing_search=False, soft_z=False,
                 memory="default", branch_prob=0.0, always_exploit=False):
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Mcts(),
                policy=policy,
                growing_search=get_grow_search_val(growing_search),
                soft_z=soft_z,
                memory=memory,
                branch_prob=branch_prob,
                always_exploit=always_exploit
            )
        )
        self.policy = policy
        self.growing_search = growing_search
        self.soft_z = soft_z
        self.memory = memory
        self.branch_prob = branch_prob
        self.always_exploit = always_exploit
        if growing_search:
            self.set_search_time(0.0)

    def new_player(self):
        return NnMctsPlayer(
            policy=self.policy, growing_search=self.growing_search, soft_z=self.soft_z,
            memory=self.memory, branch_prob=self.branch_prob, always_exploit=self.always_exploit
        )


""" ****************************************
        Minimax and Alpha Beta
    **************************************** """


class NnMinimaxPlayer(BaseExItPlayer):

    def __init__(self, use_ab=False, fixed_depth=None, policy=Policy.OFF, growing_search=False,
                 soft_z=False, growing_depth=False, memory="default", branch_prob=0.0,
                 always_exploit=False):
        if fixed_depth is not None and growing_search:
            raise Exception("Cannot have a fixed search depth and growing search timer!")
        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(fixed_depth=fixed_depth, use_ab=use_ab),
                policy=policy,
                growing_search=get_grow_search_val(growing_search),
                soft_z=soft_z,
                growing_depth=growing_depth,
                memory=memory,
                branch_prob=branch_prob,
                always_exploit=always_exploit
            )
        )
        self.use_ab = use_ab
        self.fixed_depth = fixed_depth
        self.policy = policy
        self.growing_search = growing_search
        self.soft_z = soft_z
        self.memory = memory
        self.growing_depth = growing_depth
        self.branch_prob = branch_prob
        self.always_exploit = always_exploit
        if growing_search:
            self.set_search_time(0.0)

    def new_player(self):
        return NnMinimaxPlayer(
            use_ab=self.use_ab, fixed_depth=self.fixed_depth, policy=self.policy,
            growing_search=self.growing_search, soft_z=self.soft_z,
            growing_depth=self.growing_depth, memory=self.memory, branch_prob=self.branch_prob,
            always_exploit=self.always_exploit
        )
