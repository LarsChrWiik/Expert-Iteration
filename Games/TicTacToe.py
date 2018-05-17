
from Games.Game import Game
import numpy as np
from Games.BoardFeatureExtraction import bitboard


class TicTacToe(Game):

    board = np.array([
        0, 0, 0,
        0, 0, 0,
        0, 0, 0
    ])

    winner = None

    def get_state_copy(self):
        copy = TicTacToe()
        copy.board = self.board.copy()
        return copy

    def get_legal_moves(self, player_index):
        return np.where(self.board == 0, 1, 0).nonzero()[0]

    def advance(self, player_index, action_index):
        self.board[action_index] = player_index
        self.update_game_state()

    def update_game_state(self):
        # Vertical
        if self.board[0] == self.board[1] == self.board[2] != 0:
            self.winner = self.board[0]
        if self.board[3] == self.board[4] == self.board[5] != 0:
            self.winner = self.board[3]
        if self.board[6] == self.board[7] == self.board[8] != 0:
            self.winner = self.board[6]

        # Horizontal
        if self.board[0] == self.board[3] == self.board[6] != 0:
            self.winner = self.board[0]
        if self.board[1] == self.board[4] == self.board[7] != 0:
            self.winner = self.board[1]
        if self.board[2] == self.board[5] == self.board[8] != 0:
            self.winner = self.board[2]

        # Diagonal
        if self.board[0] == self.board[4] == self.board[8] != 0:
            self.winner = self.board[0]
        if self.board[2] == self.board[4] == self.board[6] != 0:
            self.winner = self.board[2]

    def has_game_ended(self):
        return self.winner is not None

    def get_feature_vector(self, player_index):
        return bitboard(board=self.board, player_index=player_index)

    def display(self):
        char_board = ""
        for x in self.board:
            if x == 0: char_board += (' ')
            if x == 1: char_board += ('x')
            if x == 2: char_board += ('o')
        print("*** Print of TicTacToe game ***")
        print(char_board[:3])
        print(char_board[3:6])
        print(char_board[-3:])
        print()
