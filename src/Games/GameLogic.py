
from enum import Enum
import numpy as np


def bitboard(board):
    """ Generate the board feature vector from an np.array.
        This function assumes that the game contains exactly
        one type of piece and exactly two players. """
    player = 1
    opponent = 2
    player_board = np.where(board == player, 1, 0)
    opponent_board = np.where(board == opponent, 1, 0)
    return np.concatenate((player_board, opponent_board))


class BaseGame:
    """ Abstract class used to ensure that games are compatible
        with Expert Iteration algorithms (ExItAlgorithm).
        Every subclass of Game should be a perfect information game. """

    def __init__(self):
        # 0 = player1, 1 = player2. (Index of the winning player).
        self.turn = 0
        self.num_players = 2

        self.board = None
        self.num_actions = None
        self.fv_size = None

        # Indicates the winner of the game. (Index of the winning player). (-1 = draw)
        self.winner = None

    def is_game_over(self):
        return self.winner is not None

    def get_result(self, player_index):
        if player_index == self.winner:
            return GameResult.WIN
        if self.is_draw():
            return GameResult.DRAW
        return GameResult.LOSE

    def is_draw(self):
        return (self.winner == -1 or self.winner is None) and len(self.get_legal_moves()) == 0

    def next_turn(self):
        raise NotImplementedError("Please Implement this method")

    def copy(self):
        raise NotImplementedError("Please Implement this method")

    def get_legal_moves(self):
        raise NotImplementedError("Please Implement this method")

    def advance(self, a):
        raise NotImplementedError("Please Implement this method")

    def update_game_state(self):
        raise NotImplementedError("Please Implement this method")

    def get_feature_vector(self):
        raise NotImplementedError("Please Implement this method")

    def display(self):
        raise NotImplementedError("Please Implement this method")

    def board_value_to_player_index(self, rep_value):
        """ Convert Representation Value to player index """
        if rep_value == 1:
            return 0
        if rep_value == 2:
            return 1
        return -1

    def player_index_to_board_value(self, player_index):
        """ Convert player index to representation value """
        if player_index == 0:
            return 1
        if player_index == 1:
            return 2
        return 0


class GameResult(Enum):
    WIN = 1
    LOSE = -1
    DRAW = 0
    NO_RESULT = None

    def pgn_score(self):
        if self == GameResult.WIN:
            return "1"
        if self == GameResult.LOSE:
            return "0"
        if self == GameResult.DRAW:
            return "1/2"
        raise Exception("NO_RESULT")

    @staticmethod
    def get_string_results(results):
        """ Appends the result values in a string separated by ',' """
        string_results = ""
        for i, v in enumerate(results):
            string_results += str(v)
            if i != len(results) - 1:
                string_results += ","
        return string_results

    @staticmethod
    def get_players_result_list_(result):
        """ Return s the result list for each player """
        result_list = []
        for r in result:
            result_list.append(GameResult.get_result_list(r=r))
        return result_list

    @staticmethod
    def get_result_list(r: 'GameResult'):
        """ Return a list indicating the score in a list
            [x, y, z], where x = win, y = lose, and z = draw. """
        x = [0, 0, 0]
        if r == GameResult.WIN:
            x[0] += 1
        elif r == GameResult.LOSE:
            x[1] += 1
        elif r == GameResult.DRAW:
            x[2] += 1
        return x

    @staticmethod
    def get_num_outcomes():
        return 3

    @staticmethod
    def get_new_result_list():
        return [0 for _ in range(GameResult.get_num_outcomes())]
