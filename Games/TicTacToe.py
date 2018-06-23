
from Games.GameLogic import BaseGame
import numpy as np
from Games.GameLogic import bitboard


class TicTacToe(BaseGame):

    def __init__(self):
        super().__init__()
        self.num_players = 2
        self.num_actions = 9
        self.turn = 0
        self.board = np.zeros((9,), dtype=int)
        self.fv_size = len(self.get_feature_vector())

    @staticmethod
    def __rep_value_to_p_index(rep_value):
        if rep_value == 1:
            return 0
        if rep_value == 2:
            return 1
        return -1

    @staticmethod
    def __p_index_to_rep_value(player_index):
        if player_index == 0:
            return 1
        if player_index == 1:
            return 2
        return 0

    def init_new_game(self):
        return TicTacToe()

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

    def advance(self, action_index):
        if action_index is None:
            raise TypeError("action_index can not be None")
        if self.board[action_index] != 0:
            raise TypeError("Cannot place a piece on top of a piece")
        rep_value = self.__p_index_to_rep_value(player_index=self.turn)
        self.board[action_index] = rep_value
        self.update_game_state()

    def update_game_state(self):
        # Horizontal
        if self.board[0] == self.board[1] == self.board[2] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[0])
        if self.board[3] == self.board[4] == self.board[5] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[3])
        if self.board[6] == self.board[7] == self.board[8] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[6])

        # Vertical
        if self.board[0] == self.board[3] == self.board[6] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[0])
        if self.board[1] == self.board[4] == self.board[7] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[1])
        if self.board[2] == self.board[5] == self.board[8] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[2])

        # Diagonal
        if self.board[0] == self.board[4] == self.board[8] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[0])
        if self.board[2] == self.board[4] == self.board[6] != 0:
            self.winner = self.__rep_value_to_p_index(self.board[2])

        if not self.is_game_over():
            # Next turn.
            self.next_turn()

        # Is the game a draw.
        if self.is_draw():
            self.winner = -1

    def is_draw(self):
        return len(np.where(self.board == 0, 1, 0).nonzero()[0]) == 0 and \
               (self.winner is None or self.winner == -1)

    def get_feature_vector(self):
        return bitboard(board=self.board)

    def next_turn(self):
        """
        Next turn is always the other player in this game.
        """
        self.turn += 1
        if self.turn >= self.num_players:
            self.turn = 0

    def display(self):
        char_board = ""
        for x in self.board:
            if x == 0: char_board += '-'
            if x == 1: char_board += 'x'
            if x == 2: char_board += 'o'
        print("*** Print of TicTacToe game ***")
        print(char_board[:3])
        print(char_board[3:6])
        print(char_board[-3:])
        print()
