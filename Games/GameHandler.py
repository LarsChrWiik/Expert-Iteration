
from Games.BaseGame import BaseGame
from Players.BasePlayer import BasePlayer


class GameHandler:
    """
    Class object to organize a game between players.
    """

    game = None
    players = []

    def __init__(self, game: BaseGame, players: [BasePlayer]):
        """
        :param game: BaseGame object.
        :param players: List of BasePlayers, the order of the players is the turn order.
        """
        self.game = game
        self.players = players

        # Set indexes for all players in the game.
        for index, p in enumerate(players):
            p.player_index = index

    def start_game(self):
        """
        Starts the game and let the player play until the game has finished.
        """
        while not self.game.has_game_ended():
            self.players[self.game.turn].make_move(self.game)
            self.game.display()
