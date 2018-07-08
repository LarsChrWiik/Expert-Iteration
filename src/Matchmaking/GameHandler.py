
from Players.Players import BasePlayer
from Games.GameLogic import GameResult


# TODO: Add PGN notation log.
class GameHandler:
    """ Class to organize a game between players """

    def __init__(self, game_class, players: [BasePlayer], randomness):
        """ The order of "players" correspond to the players turn.
            This means that players[0] will always start the game.
            Make sure to rearrange the order of the players when playing multiple games.
            The resulted results are calculated according to the players index,
            NOT the order of the players. """
        self.game_class = game_class
        self.players = players
        self.randomness = randomness
        self.game = None

    def play_game_until_finish(self):
        """ Starts the game and let the players play until the game has finished.
            Only the apprentice is used to decide the move if the player is an ExIt player. """
        state = self.game_class()
        while not state.is_game_over():
            self.players[state.turn].move(state, randomness=self.randomness)
        self.game = state

    def get_result(self):
        """ Return a list of GameResult for each player.
            According to player index NOT the order of the players. """
        game_result = [self.game.get_result(i) for i in range(len(self.players))]
        # Order result by player index.
        result = [GameResult(None) for _ in range(len(game_result))]
        for i, p in enumerate(self.players):
            result[p.index] = game_result[i]
        return result
