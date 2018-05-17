
from Games.BaseGame import BaseGame


class BasePlayer:
    """
    Abstract class representing a Player.
    This class ensures compatibility with GameHandler.
    """

    player_index = None

    def make_move(self, game: BaseGame):
        """
        Makes the player calculate the next move.

        :param game: Base game object.
        """
        raise NotImplementedError("Please Implement this method")
