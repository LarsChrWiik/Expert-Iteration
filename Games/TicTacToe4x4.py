

from Games.GameLogic import BaseGame
import numpy as np
from Games.GameLogic import bitboard


class TicTacToe4x4(BaseGame):

    rows = 4
    columns = 4
    num_squares = columns * rows

    def __init__(self):
        super().__init__()
        self.num_players = 2
        self.turn = 0
        self.board = np.zeros((TicTacToe4x4.num_squares,), dtype=int)
        self.fv_size = TicTacToe4x4.num_squares * 2
        self.num_actions = TicTacToe4x4.num_squares
        self.in_a_row_to_win = 4

    def copy(self):
        board_copy = TicTacToe4x4()
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

        def check_in_a_row(r):
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
                        self.winner = self.board_value_to_player_index(c)
                        return
                    continue
                if c != last and c != 0:
                    # A new piece color is found, restart the counter for this color.
                    last = c
                    counter = 1

        def check_horizontal(board):
            for r in board:
                check_in_a_row(r)

        def check_diagonal(board, column_count, row_count):
            diagonals = [board.diagonal()]
            """ -> """
            for i in range(1, row_count - self.in_a_row_to_win + 1):
                diagonals.append(board.diagonal(offset=i))
            """ |
                V """
            for i in range(1, column_count - self.in_a_row_to_win + 1):
                diagonals.append(board.diagonal(offset=-i))

            for d in diagonals:
                check_in_a_row(d)

        # Convert board to matrix.
        board = np.reshape(self.board, (-1, TicTacToe4x4.columns))

        # Horizontal "-"
        check_horizontal(board)

        # Vertical "|"
        board = np.transpose(board)
        check_horizontal(board)

        # NB: Board is still transposed, but this does not matter for the checks below.
        # Diagonal "\"
        check_diagonal(board, column_count=TicTacToe4x4.columns, row_count=TicTacToe4x4.rows)

        # Diagonal "/"
        board = np.rot90(board)
        check_diagonal(board, column_count=TicTacToe4x4.rows, row_count=TicTacToe4x4.columns)

        self.next_turn()

        # Is the game a draw.
        if self.is_draw():
            self.winner = -1

    def get_feature_vector(self):
        return bitboard(self.board)

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
        c = TicTacToe4x4.columns
        for r in range(c):
            print(char_board[r*c:r*c + c])
        print()