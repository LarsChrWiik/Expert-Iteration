
from Games.GameLogic import BaseGame
from ExIt.Apprentice.RandomPredictor import RandomPredictor
from ExIt.Apprentice.Nn import Nn
from ExIt.Expert.Minimax import Minimax
from ExIt.Expert.Mcts import Mcts
from ExIt.ExpertIteration import ExpertIteration
from Players.BasePlayers import BasePlayer, BaseExItPlayer
from ExIt.Policy import Policy


def assert_kwargs(main_kwargs, kwargs):
    for key, value in kwargs.items():
        if key not in main_kwargs:
            raise Exception("Unknown keyword:", key, "with value", kwargs.get(key))


def assert_new_kwargs(kwargs):
    fixed_depth = kwargs.get("fixed_depth")
    growing_search_time = kwargs.get("growing_search_time")
    growing_depth = kwargs.get("growing_depth")

    # Check if fixed depth is bool.
    if isinstance(fixed_depth, bool) and fixed_depth is not None:
        raise Exception("fixed_depth much to be an integer")

    # growing_search_time + growing_depth
    if growing_search_time and growing_depth:
        raise Exception("Cannot have a fixed search depth and growing depth.")

    # growing_search_time + fixed_depth
    if growing_search_time and fixed_depth:
        raise Exception("Cannot have a growing search time and fixed depth.")

    # growing_depth + fixed_depth
    if growing_depth and fixed_depth:
        raise Exception("Cannot have a growing depth and fixed depth.")


""" ****************************************
        Random Player
    **************************************** """


class RandomPlayer(BasePlayer):
    """ Player that plays random moves """

    def __init__(self):
        super().__init__()
        self.__name__ = type(self).__name__

    def move(self, state: BaseGame, randomness=False, verbose=False):
        return self.move_random(state), None, None


""" ****************************************
        Brute-force player
    **************************************** """


class BruteForcePlayer(BasePlayer):
    """ Static Minimax Player with a fixed depth """

    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.minimax = Minimax(fixed_depth=depth)
        self.predictor = RandomPredictor()
        self.__name__ = type(self).__name__ + "_depth-" + str(depth)

    def move(self, state: BaseGame, verbose=False):
        _, a, v = self.minimax.search(
            state=state,
            predictor=self.predictor,
            always_exploit=True
        )
        state.advance(a)
        return a


""" ****************************************
        MCTS Player
    **************************************** """


class NnMctsPlayer(BaseExItPlayer):

    default_kwargs = {
        "policy": Policy.OFF,
        "growing_search_time": False,
        "min_growing_time": None,
        "soft_z": False,
        "memory": "default",
        "branch_prob": 0.0,
        "always_exploit": False
    }

    def __init__(self, **kwargs):
        self.kwargs = self.default_kwargs.copy()
        assert_kwargs(self.kwargs, kwargs)
        self.kwargs.update(kwargs)
        assert_new_kwargs(self.kwargs)

        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Mcts(),
                **self.kwargs
            )
        )

    def new(self):
        return NnMctsPlayer(**self.kwargs)


""" ****************************************
        Minimax / Alpha Beta Player
    **************************************** """


class NnMinimaxPlayer(BaseExItPlayer):

    default_kwargs = {
        "use_ab": False,
        "policy": Policy.OFF,
        "fixed_depth": None,
        "growing_search_time": False,
        "min_growing_time": None,
        "growing_depth": False,
        "soft_z": False,
        "memory": "default",
        "branch_prob": 0.0,
        "always_exploit": False
    }

    def __init__(self, **kwargs):
        self.kwargs = self.default_kwargs.copy()
        assert_kwargs(self.kwargs, kwargs)
        self.kwargs.update(kwargs)
        assert_new_kwargs(self.kwargs)

        super().__init__(
            ex_it_algorithm=ExpertIteration(
                apprentice=Nn(),
                expert=Minimax(
                    fixed_depth=self.kwargs.get("fixed_depth"),
                    use_ab=self.kwargs.get("use_ab")
                ),
                **self.kwargs
            )
        )

    def new(self):
        return NnMinimaxPlayer(**self.kwargs)
