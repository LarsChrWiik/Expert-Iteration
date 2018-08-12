
from Games.GameLogic import BaseGame
import numpy as np
from Games.GameLogic import bitboard


class Othello(BaseGame):

    rows = 6
    columns = 6
    num_squares = columns * rows
    num_players = 2
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        ( 0, -1),          ( 0, 1),
        ( 1, -1), ( 1, 0), ( 1, 1)
    ]

    def __init__(self):
        super().__init__()
        self.turn = 0
        self.board = np.zeros((Othello.num_squares,), dtype=int)
        self.board[Othello.get_board_index(2, 2)] = 1
        self.board[Othello.get_board_index(2, 3)] = 2
        self.board[Othello.get_board_index(3, 2)] = 2
        self.board[Othello.get_board_index(3, 3)] = 1
        self.fv_size = Othello.num_squares * 2
        self.num_actions = Othello.num_squares
        self.in_a_row_to_win = 6

    def copy(self):
        board_copy = Othello()
        board_copy.board = self.board.copy()
        board_copy.winner = self.winner
        board_copy.turn = self.turn
        return board_copy

    def get_legal_moves(self):
        """ Return a list of the possible action indexes """
        if self.is_game_over():
            return []

        legal_actions = []
        for i in range(Othello.rows):
            for j in range(Othello.columns):
                # For all squares on the board.
                if self.get_board_square(i, j) != 0:
                    continue

                for d in Othello.directions:
                    # Check all directions.
                    if self.piece_check_direction(i, j, d, turn=self.turn):
                        legal_actions.append(i*Othello.columns + j)
        legal_actions = list(set(legal_actions))
        return np.array(legal_actions)

    def piece_check_direction(self, i, j, d, turn):
        color = self.player_index_to_board_value(turn)
        r, c = d
        i_new, j_new = i + r, j + c

        # Make sure the next piece is inside the board.
        if not Othello.is_inside_board(i_new, j_new):
            return False
        # Make sure the next piece is opponent's piece.
        if self.get_board_square(i_new, j_new) in [0, color]:
            return False

        i_new, j_new = i_new + r, j_new + c

        while Othello.is_inside_board(i_new, j_new):
            if self.get_board_square(i_new, j_new) == 0:
                return False
            if self.board[Othello.get_board_index(i_new, j_new)] == color:
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

    @staticmethod
    def is_inside_board(i, j):
        return 0 <= i < Othello.rows and 0 <= j < Othello.columns

    def get_board_square(self, i, j):
        return self.board[Othello.get_board_index(i, j)]

    @staticmethod
    def get_board_index(i, j):
        return i * Othello.columns + j

    def update_game_state(self, a):
        i = int(a / Othello.columns)
        j = a % Othello.columns
        color = self.get_board_square(i, j)

        for d in Othello.directions:
            r, c = d
            i_new, j_new = i + r, j + c
            if self.piece_check_direction(i, j, d, turn=self.turn):
                while not self.board[Othello.get_board_index(i_new, j_new)] == color:
                    self.board[Othello.get_board_index(i_new, j_new)] = color
                    i_new, j_new = i_new + r, j_new + c

        self.next_turn()

        # Is the game a draw.
        if self.is_draw():
            self.winner = -1

    def get_feature_vector(self):
        return bitboard(self.board)

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
        c = Othello.columns
        for r in range(c):
            print(char_board[r*c:r*c + c])
        print()
