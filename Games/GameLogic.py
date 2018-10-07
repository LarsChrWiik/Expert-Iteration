
from enum import Enum
import numpy as np
from copy import deepcopy


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

    def get_augmentations(self, s_array, pi_array, v_array):
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

    def is_inside_board(self, i, j):
        return 0 <= i < self.rows and 0 <= j < self.columns

    def get_board_square(self, i, j):
        return self.board[self.get_board_index(i, j)]

    def get_board_index(self, i, j):
        return i * self.columns + j

    def matrix(self, array):
        return array.reshape(self.rows, self.columns)

    # Augmentations.

    def rotate_90(self, array, k=1):
        return np.rot90(self.matrix(array), k).flatten()

    def transpose_array(self, array):
        return np.transpose(self.matrix(array)).flatten()

    def transpose_s(self, s):
        return self.__s_function(s, self.transpose_array)

    @staticmethod
    def map_add(arrays, *functions):
        array_new = []
        for f in functions:
            array_new += list(map(f, arrays))
        return array_new

    def __s_function(self, s, function=None, *arg):
        p1 = function(s[:len(self.board)], *arg)
        p2 = function(s[len(self.board):], *arg)
        return np.concatenate([p1, p2])

    def aug_rotate_90(self, array):
        return np.rot90(self.matrix(array)).flatten()

    def aug_rotate_180(self, array):
        return np.rot90(self.matrix(array), 2).flatten()

    def aug_rotate_270(self, array):
        return np.rot90(self.matrix(array), 3).flatten()

    def aug_flip_vertical(self, array):
        return np.flip(self.matrix(array), 1).flatten()

    def aug_flip_horizontal(self, array):
        return np.flip(self.matrix(array), 0).flatten()

    def aug_rotate_90_s(self, s):
        return self.__s_function(s, self.rotate_90, 1)

    def aug_rotate_180_s(self, s):
        return self.__s_function(s, self.rotate_90, 2)

    def aug_rotate_270_s(self, s):
        return self.__s_function(s, self.rotate_90, 3)

    def aug_flip_horizontal_s(self, s):
        return self.__s_function(s, self.aug_flip_horizontal)

    def aug_flip_vertical_s(self, s):
        return self.__s_function(s, self.aug_flip_vertical)

    def get_all_augmentations(self, s_array, pi_array, v_array):

        s_array_new, pi_array_new, v_array_new = [], [], []

        # S
        s_array_new += list(map(lambda x : x, s_array))
        s_array_new += self.map_add(
            s_array,
            self.aug_rotate_90_s,
            self.aug_rotate_180_s,
            self.aug_rotate_270_s
        )
        s_array_transposed = list(map(self.transpose_s, s_array))
        s_array_new += s_array_transposed.copy()
        s_array_new += self.map_add(
            s_array_transposed,
            self.aug_rotate_90_s,
            self.aug_rotate_180_s,
            self.aug_rotate_270_s
        )

        # PI
        pi_array_new += list(map(lambda x : x, pi_array))
        pi_array_new += self.map_add(
            pi_array,
            self.aug_rotate_90,
            self.aug_rotate_180,
            self.aug_rotate_270
        )
        pi_array_transposed = list(map(self.transpose_array, pi_array))
        pi_array_new += pi_array_transposed.copy()
        pi_array_new += self.map_add(
            pi_array_transposed,
            self.aug_rotate_90,
            self.aug_rotate_180,
            self.aug_rotate_270
        )

        # V
        for _ in range(8):
            v_array_new += v_array

        return s_array_new, pi_array_new, v_array_new


class InARowGameSquareBoard(BaseGameSquareBoard):

    def __init__(self):
        super().__init__()
        self.in_a_row_to_win = None

    def check_in_a_row(self, r):
        counter = 0
        last = -1
        for c in r:
            if c == 0:
                # This field is not taken by any player.
                counter = 0
                last = -1
                continue
            if c == last:
                # The same piece color as last time, check if this player has won.
                counter += 1
                if counter == self.in_a_row_to_win:
                    # WINNER!
                    self.winner = self.board_value_to_player_index(c)
                    return
                continue
            if c != last and c != 0:
                # A new piece color is found, restart the counter for this color.
                last = c
                counter = 1

    def check_horizontal(self, board):
        for r in board:
            self.check_in_a_row(r)

    def check_diagonal(self, board, column_count, row_count):
        diagonals = [board.diagonal()]
        """ -> """
        for i in range(1, row_count - self.in_a_row_to_win + 1):
            diagonals.append(board.diagonal(offset=i))
        """ |
            V """
        for i in range(1, column_count - self.in_a_row_to_win + 1):
            diagonals.append(board.diagonal(offset=-i))

        for d in diagonals:
            self.check_in_a_row(d)

    def update_in_a_row_game(self):
        # Convert board to matrix.
        board = np.reshape(self.board, (-1, self.columns))
        # Horizontal "-"
        self.check_horizontal(board)
        # Vertical "|"
        board = np.transpose(board)
        self.check_horizontal(board)
        # NB: Board is still transposed, but this does not matter for the checks below.
        # Diagonal "\"
        self.check_diagonal(board, column_count=self.columns, row_count=self.rows)
        # Diagonal "/"
        board = np.rot90(board)
        self.check_diagonal(board, column_count=self.rows, row_count=self.columns)


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
