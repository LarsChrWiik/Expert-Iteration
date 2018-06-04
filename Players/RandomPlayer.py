
from Players.Base import BasePlayer
from Games.BaseGame import BaseGame
import random


class RandomPlayer(BasePlayer):
    """ Player that plays random moves """

    def move(self, game: BaseGame):
        legal_moves = game.get_legal_moves(self.player_index)
        action_index = random.choice(legal_moves)
        game.advance(action_index=action_index)
