
from Games.BaseGame import BaseGame
import numpy as np
from Games.BoardFeatureExtraction import bitboard


class TicTacToe(BaseGame):
    """
    The game of TicTacToe.
    """

    num_players = 2
    num_actions = 9

    @staticmethod
    def __rep_value_to_p_index(rep_value):
        if rep_value == 1: return 0
        if rep_value == 2: return 1
        return -1

    @staticmethod
    def __p_index_to_rep_value(player_index):
        if player_index == 0: return 1
        if player_index == 1: return 2
        return 0

    def __init__(self, turn=0):
        self.turn = turn
        self.board = np.zeros((9,), dtype=int)

    def get_state_copy(self):
        board_copy = TicTacToe()
        board_copy.board = self.board.copy()
        board_copy.winner = self.winner
        board_copy.turn = self.turn
        return board_copy

    def get_legal_moves(self, player_index):
        """
        :param player_index: Not used in this game.
        :return: list of possible move indexes.
        """
        if self.has_finished():
            return []
        return np.where(self.board == 0, 1, 0).nonzero()[0]

    def advance(self, action_index):
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

        # Next turn.
        self.next_turn()

        # Is the game a draw.
        if self.winner is None and self.is_draw():
            self.winner = -1

    def is_draw(self):
        return len(np.where(self.board == 0, 1, 0).nonzero()[0]) == 0 and \
               (self.winner is None or self.winner == -1)

    def get_feature_vector(self, player_index):
        return bitboard(board=self.board, player_index=player_index)

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
