
from Games.GameLogic import BaseGame
import numpy as np
from Games.GameLogic import bitboard


class TicTacToe(BaseGame):

    num_squares = 9

    def __init__(self):
        super().__init__()
        self.board = np.zeros((TicTacToe.num_squares,), dtype=int)
        self.fv_size = TicTacToe.num_squares * 2
        self.num_actions = TicTacToe.num_squares

    def copy(self):
        board_copy = TicTacToe()
        board_copy.board = self.board.copy()
        board_copy.winner = self.winner
        board_copy.turn = self.turn
        return board_copy

    def get_legal_moves(self):
        """ Return a list of the possible action indexes """
        if self.is_game_over():
            return []
        return np.where(self.board == 0, 1, 0).nonzero()[0]

    def advance(self, a):
        if self.winner is not None:
            raise Exception("Cannot advance when game is over")
        if a is None:
            raise Exception("action_index can not be None")
        if self.board[a] != 0:
            raise Exception("Cannot place a piece on top of a piece")
        if a >= self.num_actions or a < 0:
            raise Exception("Action is not legal")
        board_value = self.player_index_to_board_value(player_index=self.turn)
        self.board[a] = board_value
        self.update_game_state()

    def update_game_state(self):
        # Horizontal "-"
        if self.board[0] == self.board[1] == self.board[2] != 0:
            self.winner = self.board_value_to_player_index(self.board[0])
        if self.board[3] == self.board[4] == self.board[5] != 0:
            self.winner = self.board_value_to_player_index(self.board[3])
        if self.board[6] == self.board[7] == self.board[8] != 0:
            self.winner = self.board_value_to_player_index(self.board[6])

        # Vertical "|"
        if self.board[0] == self.board[3] == self.board[6] != 0:
            self.winner = self.board_value_to_player_index(self.board[0])
        if self.board[1] == self.board[4] == self.board[7] != 0:
            self.winner = self.board_value_to_player_index(self.board[1])
        if self.board[2] == self.board[5] == self.board[8] != 0:
            self.winner = self.board_value_to_player_index(self.board[2])

        # Diagonal /
        if self.board[0] == self.board[4] == self.board[8] != 0:
            self.winner = self.board_value_to_player_index(self.board[0])
        if self.board[2] == self.board[4] == self.board[6] != 0:
            self.winner = self.board_value_to_player_index(self.board[2])

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
        print(char_board[:3])
        print(char_board[3:6])
        print(char_board[-3:])
        print()
