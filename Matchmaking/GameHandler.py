
from Players.Players import BasePlayer
from Games.GameLogic import GameResult, BaseGame


def match(game_class, players, randomness):
    """ Starts a new game match between players and returns the result.
        Return a list of GameResult for each player according to Game Index. """
    game_handler = GameHandler(game_class, players, randomness)
    game_handler.play_game_until_finish()
    return game_handler.get_result()


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
        self.last_turn = None
        self.move_switch_counter = 1
        self.move_text = "NONE"
        self.result_text = "NONE"

    def update_movetext_result(self, state: BaseGame):
        self.result_text = state.get_result(0).pgn_score() + "-" + state.get_result(1).pgn_score()
        self.move_text += " " + self.result_text

    def update_movetext(self, a, turn, p, v):
        if turn == 0 or turn == self.last_turn:
            self.move_text += str(self.move_switch_counter) + ".\n"
            self.move_switch_counter += 1
        if p is not None:
            p = [str(round(p_val, 2)) for p_val in p]
            p_text = "{" + ', '.join(p) + "}"
            v_text = str(round(v, 3))
        else:
            p_text = "None"
            v_text = "None"
        self.move_text += "   a" + str(a) + " | p=" + p_text + ", v=" + v_text + "\n"

    def play_game_until_finish(self):
        """ Starts the game and let the players play until the game has finished.
            Only the apprentice is used to decide the move if the player is an ExIt player. """
        self.move_text = ""
        self.result_text = ""
        self.last_turn = None
        self.move_switch_counter = 1
        state = self.game_class()
        while not state.is_game_over():
            turn = state.turn
            a, p, v = self.players[state.turn].move(state, randomness=self.randomness)
            self.update_movetext(a, turn, p, v)
            self.last_turn = turn
        self.update_movetext_result(state)
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
