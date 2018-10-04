
from Games.GameLogic import InARowGameSquareBoard
import numpy as np
from Games.GameLogic import bitboard


class ConnectSix(InARowGameSquareBoard):

    default_kwargs = {
        "rows": 10,
        "columns": 10,
        "in_a_row_to_win": 6
    }

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = self.default_kwargs.copy()
        self.kwargs.update(kwargs)
        self.rows = self.kwargs.get("rows")
        self.columns = self.kwargs.get("columns")
        self.in_a_row_to_win = self.kwargs.get("in_a_row_to_win")

        self.num_squares = self.columns * self.rows
        self.board = np.zeros((self.num_squares,), dtype=int)
        self.fv_size = self.num_squares * 2
        self.num_actions = self.num_squares

        self.kwargs = kwargs

        self.__name__ = "ConnectSix" + str(self.rows) + "x" + str(self.columns)

    def new(self):
        return ConnectSix(**self.kwargs)

    def copy(self):
        board_copy = ConnectSix(**self.kwargs)
        board_copy.board = self.board.copy()
        board_copy.winner = self.winner
        board_copy.turn = self.turn
        return board_copy

    def get_legal_moves(self):
        """ Return a list of the possible action indexes """
        if self.is_game_over():
            return []
        return np.where(self.board[:self.num_actions] == 0, 1, 0).nonzero()[0]

    def advance(self, a):
        if self.winner is not None:
            raise Exception("Cannot advance when game is over")
        if a is None:
            raise Exception("action_index can not be None")
        if self.board[a] != 0:
            raise Exception("This column is full")
        if a >= self.num_actions or a < 0:
            raise Exception("Action is not legal")

        board_value = self.player_index_to_board_value(player_index=self.turn)
        self.board[a] = board_value
        self.update_game_state()

    def update_game_state(self):
        self.update_in_a_row_game()
        self.next_turn()
        # Is the game a draw.
        if self.is_draw():
            self.winner = -1

    def get_augmentations(self, s_array, pi_array, v_array):
        return self.get_all_augmentations(s_array, pi_array, v_array)

    def get_feature_vector(self):
        return bitboard(self.board, self.player_index_to_board_value(self.turn))

    def next_turn(self):
        """ Next turn is always the other player in this game """
        self.turn += 1
        if self.turn >= self.num_players:
            self.turn = 0

    def display(self):
        char_board = ""
        for x in self.board:
            if x == 0: char_board += '-'
            if x == 1: char_board += 'x'
            if x == 2: char_board += 'o'
        print("*** Print of " + str(type(self).__name__) + " game ***")
        c = self.columns
        for r in range(c):
            print(char_board[r*c:r*c + c])
        print()
