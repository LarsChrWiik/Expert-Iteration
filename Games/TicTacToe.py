
from Games.BaseGame import BaseGame
import numpy as np
from Games.BoardFeatureExtraction import bitboard


class TicTacToe(BaseGame):
    """
    The game of TicTacToe.
    """

    num_players = 2
    num_actions = 9
    num_rotations = 4
    fv_size = 18

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

        # TODO: Used for testing.
        """
        self.advance(1)
        self.advance(4)
        self.advance(3)
        self.advance(5)
        """

    def init_new_game(self):
        return TicTacToe()

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

    def rotate_fv(self, fv: np.array):
        fv_new = np.zeros(fv.size)
        fv_new[6] = fv[0];  fv_new[3] = fv[1];  fv_new[0] = fv[2]
        fv_new[7] = fv[3];  fv_new[4] = fv[4];  fv_new[1] = fv[5]
        fv_new[8] = fv[6];  fv_new[5] = fv[7];  fv_new[2] = fv[8]

        fv_new[6+9] = fv[0+9]
        fv_new[3+9] = fv[1+9]
        fv_new[0+9] = fv[2+9]
        fv_new[7+9] = fv[3+9]
        fv_new[4+9] = fv[4+9]
        fv_new[1+9] = fv[5+9]
        fv_new[8+9] = fv[6+9]
        fv_new[5+9] = fv[7+9]
        fv_new[2+9] = fv[8+9]
        return fv_new

    def rotate_pi(self, pi):
        pi_new = np.zeros(pi.size)
        pi_new[6] = pi[0]
        pi_new[3] = pi[1]
        pi_new[0] = pi[2]
        pi_new[7] = pi[3]
        pi_new[4] = pi[4]
        pi_new[1] = pi[5]
        pi_new[8] = pi[6]
        pi_new[5] = pi[7]
        pi_new[2] = pi[8]
        return pi_new

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
