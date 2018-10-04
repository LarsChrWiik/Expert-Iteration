
from Games.GameLogic import InARowGameSquareBoard
import numpy as np
from Games.GameLogic import bitboard


class ConnectFour(InARowGameSquareBoard):

    default_kwargs = {
        "rows": 6,
        "columns": 7,
        "in_a_row_to_win": 4
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
        self.num_actions = self.columns
        self.in_a_row_to_win = 4

        self.kwargs = kwargs

        self.__name__ = "ConnectFour" + str(self.rows) + "x" + str(self.columns)

    def new(self):
        return ConnectFour(**self.kwargs)

    def copy(self):
        board_copy = ConnectFour(**self.kwargs)
        board_copy.board = self.board.copy()
        board_copy.winner = self.winner
        board_copy.turn = self.turn
        return board_copy

    def get_legal_moves(self):
        """ Return a list of the possible action indexes """
        if self.is_game_over():
            return []
        return np.where(self.board[:self.columns] == 0, 1, 0).nonzero()[0]

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
        # Start from the end because the piece falls down.
        reversed_a = self.columns - a
        while True:
            if self.board[-reversed_a] == 0:
                self.board[-reversed_a] = board_value
                break
            else:
                # This place is take, check the place above next time.
                reversed_a += self.columns
        self.update_game_state()

    def update_game_state(self):
        self.update_in_a_row_game()
        self.next_turn()
        # Is the game a draw.
        if self.is_draw():
            self.winner = -1

    def get_augmentations(self, s_array, pi_array, v_array):
        s_array_new, pi_array_new, v_array_new = [], [], []

        # S
        s_array_new += list(map(lambda x: x, s_array))
        s_array_new += self.map_add(
            s_array,
            self.aug_flip_vertical_s
        )

        # PI
        pi_array_new += list(map(lambda x: x, pi_array))
        pi_array_new += self.map_add(
            pi_array,
            lambda x: np.flip(x, 0),
        )

        # V
        for _ in range(2):
            v_array_new += v_array

        return s_array_new, pi_array_new, v_array_new

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
