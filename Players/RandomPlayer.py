
from Players.BasePlayer import BasePlayer
from Games.BaseGame import BaseGame
import random


class RandomPlayer(BasePlayer):
    """
    Player that plays random moves.
    """

    def make_move(self, game: BaseGame):
        legal_moves = game.get_legal_moves(self.player_index)
        action_index = random.choice(legal_moves)
        game.advance(
            player_index=self.player_index,
            action_index=action_index
        )
