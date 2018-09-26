
from enum import Enum
import numpy as np


def bitboard(board, player_index):
    """ Generate the board feature vector from an np.array.
        This function assumes that the game contains exactly
        one type of piece and exactly two players. """
    player = player_index
    opponent = player_index + 1
    if opponent > 2:
        opponent = 1
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

    def new(self):
        raise NotImplementedError("Please Implement this method")

    def is_game_over(self):
        return self.winner is not None

    def get_result(self, player_index):
        if player_index == self.winner:
            return GameResult.WIN
        if self.is_draw():
            return GameResult.DRAW
        return GameResult.LOSE

    def is_draw(self):
        if self.winner == -1:
            return True
        if self.winner == 0 or self.winner == 1:
            return False
        # This must be reimplemented if a player can skip a move.
        # This should then be tested for each player.
        return len(self.get_legal_moves()) == 0

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

    def add_augmentations(self, s_array, pi_array, v_array):
        return s_array, pi_array, v_array


class BaseGameSquareBoard(BaseGame):

    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1)
    ]

    def __init__(self):
        super().__init__()
        self.rows = None
        self.columns = None

    def convert_1d_to_2d_board(self, array):
        board_2d = []
        for r in range(self.rows):
            row = []
            for c in range(self.columns):
                row.append(array[r*self.columns+c])
            board_2d.append(row)
        return board_2d

    def convert_2d_to_1d_board(self, array):
        board_1d = []
        for r in range(self.rows):
            for c in range(self.columns):
                board_1d.append(array[r][c])
        return board_1d

    @staticmethod
    def rotate_clockwise_2D(array_2d):
        return list(list(x)[::-1] for x in zip(*array_2d))

    def rotate_clockwise_1D(self, x):
        x = self.convert_1d_to_2d_board(x)
        x = self.rotate_clockwise_2D(x)
        return self.convert_2d_to_1d_board(x)

    def rotate_fv_clockwise(self, s_array):
        p1 = self.rotate_clockwise_1D(s_array[:len(self.board)])
        p2 = self.rotate_clockwise_1D(s_array[len(self.board):])
        return p1 + p2

    def is_inside_board(self, i, j):
        return 0 <= i < self.rows and 0 <= j < self.columns

    def get_board_square(self, i, j):
        return self.board[self.get_board_index(i, j)]

    def get_board_index(self, i, j):
        return i * self.columns + j

    @staticmethod
    def transpose_2D(matrix):
        return list(map(list, zip(*matrix)))

    def transpose_1D(self, x):
        x = self.convert_1d_to_2d_board(x)
        x = self.transpose_2D(x)
        return self.convert_2d_to_1d_board(x)

    def transpose_fv(self, fv):
        p1 = self.transpose_1D(fv[:len(self.board)])
        p2 = self.transpose_1D(fv[len(self.board):])
        return p1 + p2

    def new(self):
        super().new()

    def next_turn(self):
        super().next_turn()

    def copy(self):
        super().copy()

    def get_legal_moves(self):
        super().get_legal_moves()

    def advance(self, a):
        super().advance(a)

    def update_game_state(self):
        super().update_game_state()

    def get_feature_vector(self):
        super().get_feature_vector()

    def display(self):
        super().display()


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
