
from Games.GameLogic import BaseGame
import numpy as np
from Games.GameLogic import bitboard


class Othello(BaseGame):

    kwargs = {
        "rows": 8,
        "columns": 8
    }

    num_players = 2
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs.update(kwargs)
        self.rows = self.kwargs.get("rows")
        self.columns = self.kwargs.get("columns")

        self.num_squares = self.columns * self.rows
        self.board = np.zeros((self.num_squares,), dtype=int)
        self.fv_size = self.num_squares * 2
        self.num_actions = self.num_squares
        self.turn = 0

        self.place_initial_pieces()
        self.kwargs = kwargs

        self.__name__ = "Othello" + str(self.rows) + "x" + str(self.columns)

    def new(self):
        return Othello(**self.kwargs)

    def copy(self):
        board_copy = Othello(**self.kwargs)
        board_copy.board = self.board.copy()
        board_copy.winner = self.winner
        board_copy.turn = self.turn
        return board_copy

    def place_initial_pieces(self):
        r = int(self.rows / 2) - 1
        c = int(self.columns / 2) - 1
        self.board[self.get_board_index(r, c)] = 1
        self.board[self.get_board_index(r, c+1)] = 2
        self.board[self.get_board_index(r+1, c)] = 2
        self.board[self.get_board_index(r+1, c+1)] = 1

    def get_legal_moves(self):
        return self.get_legal_moves_2(self.turn)

    def get_legal_moves_2(self, turn):
        """ Return a list of the possible action indexes """
        if self.is_game_over():
            return []

        legal_actions = []
        for i in range(self.rows):
            for j in range(self.columns):
                # For all squares on the board.
                if self.get_board_square(i, j) != 0:
                    continue

                for d in Othello.directions:
                    # Check all directions.
                    if self.piece_check_direction(i, j, d, turn=turn):
                        legal_actions.append(i * self.columns + j)
        legal_actions = list(set(legal_actions))
        return np.array(legal_actions)

    def piece_check_direction(self, i, j, d, turn):
        color = self.player_index_to_board_value(turn)
        r, c = d
        i_new, j_new = i + r, j + c

        # Make sure the next piece is inside the board.
        if not self.is_inside_board(i_new, j_new):
            return False
        # Make sure the next piece is opponent's piece.
        if self.get_board_square(i_new, j_new) in [0, color]:
            return False

        i_new, j_new = i_new + r, j_new + c

        while self.is_inside_board(i_new, j_new):
            if self.get_board_square(i_new, j_new) == 0:
                return False
            if self.board[self.get_board_index(i_new, j_new)] == color:
                return True
            i_new, j_new = i_new + r, j_new + c
        return False

    def advance(self, a):
        if self.winner is not None:
            raise Exception("Cannot advance when game is over")
        if a is None:
            raise Exception("action_index can not be None")
        if self.board[a] != 0:
            raise Exception("This column is full")
        if a >= self.num_actions or a < 0:
            raise Exception("Action is not legal")
        if a not in self.get_legal_moves():
            raise Exception("Action is not legal according to get_legal_moves function")

        board_value = self.player_index_to_board_value(player_index=self.turn)
        self.board[a] = board_value
        self.update_game_state(a)

    def is_inside_board(self, i, j):
        return 0 <= i < self.rows and 0 <= j < self.columns

    def get_board_square(self, i, j):
        return self.board[self.get_board_index(i, j)]

    def get_board_index(self, i, j):
        return i * self.columns + j

    def update_game_state(self, a):
        i = int(a / self.columns)
        j = a % self.columns
        color = self.get_board_square(i, j)

        for d in Othello.directions:
            r, c = d
            i_new, j_new = i + r, j + c
            if self.piece_check_direction(i, j, d, turn=self.turn):
                while not self.board[self.get_board_index(i_new, j_new)] == color:
                    self.board[self.get_board_index(i_new, j_new)] = color
                    i_new, j_new = i_new + r, j_new + c

        self.next_turn()

        if self.is_draw():
            self.winner = -1

    def get_feature_vector(self):
        return bitboard(self.board)

    def is_draw(self):
        if self.winner == -1:
            return True
        if self.winner == 0 or self.winner == 1:
            return False
        # This must be reimplemented if a player can skip a move.
        # This should then be tested for each player.
        count_0 = len([x for x in self.board if x == 1])
        count_1 = len([x for x in self.board if x == 2])
        return len(self.get_legal_moves_2(self.turn)) == 0 and \
               len(self.get_legal_moves_2(Othello.other_turn(self.turn))) == 0 and \
               count_0 == count_1

    @staticmethod
    def other_turn(t):
        return 0 if t == 1 else 1

    def next_turn(self):
        self.turn += 1
        if self.turn >= Othello.num_players:
            self.turn = 0

        if len(self.get_legal_moves()) == 0:
            self.turn += 1
            if self.turn >= Othello.num_players:
                self.turn = 0
            if len(self.get_legal_moves()) == 0:
                # GAME STOPS. Declare winner.
                self.declare_winner()

    def declare_winner(self):
        count_0 = len([x for x in self.board if x == 1])
        count_1 = len([x for x in self.board if x == 2])
        if count_0 == count_1:
            self.winner = -1
        if count_0 > count_1:
            self.winner = self.board_value_to_player_index(1)
        if count_1 > count_0:
            self.winner = self.board_value_to_player_index(2)

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
