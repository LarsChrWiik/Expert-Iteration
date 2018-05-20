
from Players.BasePlayer import BasePlayer
from Games.BaseGame import BaseGame
from ExIt.ExItAlgorithm import ExItAlgorithm


class BaseExItPlayer(BasePlayer):
    """
    Player that is able to improve its strategy using Expert Iteration.
    """

    exIt_algorithm = None

    def __init__(self, exIt_algorithm: ExItAlgorithm):
        self.exIt_algorithm = exIt_algorithm

    def start_expert_iteration(self, game: BaseGame):
        """
        Starts Expert Iteration to master the given game.
        This process might take some time.
        """
        self.exIt_algorithm.start_expert_iteration(game_original=game)

    def make_move(self, game: BaseGame):
        """
        This is used after the Expert Iteration has completed.

        :param game: BaseGame object.
        """
        action_index = self.exIt_algorithm.calculate_best_action(game)
        game.advance(action_index=action_index)
