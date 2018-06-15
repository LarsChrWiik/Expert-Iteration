
from Games.BaseGame import BaseGame
from Players.Players import BasePlayer


class GameHandler:
    """ Class to organize games between players """

    def __init__(self, game: BaseGame, players: [BasePlayer]):
        self.game = game
        self.players = players

        # Index all players in the game.
        for index, p in enumerate(players):
            p.player_index = index

    def start_game(self):
        """ Starts the game and let the players play until the game has finished """
        while not self.game.is_game_over():
            self.players[self.game.turn].make_expert_move(self.game)
