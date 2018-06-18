
from enum import Enum
import numpy as np


def bitboard(board, player_index):
    """ Generate the board feature vector from an np.array.
        This function assumes that the game contains exactly
        one type of piece and exactly two players. """
    player_representation = player_index + 1
    opponent_representation = 2 if player_representation == 1 else 1
    player = np.where(board == player_representation, 1, 0)
    opponent = np.where(board == opponent_representation, 1, 0)
    return np.concatenate((player, opponent))


class BaseGame:
    """ Abstract class used to ensure that games are compatible
        with Expert Iteration algorithms (ExItAlgorithm).
        Every subclass of Game should be a perfect information game. """

    win_reward = 1
    lose_reward = -1
    draw_reward = 0

    board = None
    num_players = None
    num_actions = None
    num_rotations = None
    fv_size = None

    # Indicates the winner of the game. (Index of the winning player). (-1 = draw)
    winner = None

    # 0 = player1, 1 = player2. (Index of the winning player).
    turn = None

    def is_game_over(self):
        return self.winner is not None

    def get_reward(self, player_index):
        if player_index == self.winner:
            return self.win_reward
        elif self.is_draw():
            return self.draw_reward
        return self.lose_reward

    def get_result(self, player_index):
        if player_index == self.winner:
            return GameResult.WIN
        if self.is_draw():
            return GameResult.DRAW
        return GameResult.LOSE

    def is_draw(self):
        raise NotImplementedError("Please Implement this method")

    def next_turn(self):
        raise NotImplementedError("Please Implement this method")

    def copy(self):
        raise NotImplementedError("Please Implement this method")

    def init_new_game(self):
        raise NotImplementedError("Please Implement this method")

    def get_possible_actions(self):
        raise NotImplementedError("Please Implement this method")

    def advance(self, action_index):
        raise NotImplementedError("Please Implement this method")

    def update_game_state(self):
        raise NotImplementedError("Please Implement this method")

    def get_feature_vector(self, player_index):
        raise NotImplementedError("Please Implement this method")

    def display(self):
        raise NotImplementedError("Please Implement this method")

    def rotate_fv(self, fv):
        raise NotImplementedError("Please Implement this method")

    def rotate_pi(self, pi):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def __rep_value_to_p_index(rep_value):
        """ Convert Representation Value to player index. """
        raise NotImplementedError("Please Implement this method")


class GameResult(Enum):
    WIN = 1
    LOSE = -1
    DRAW = 0
    NO_RESULT = None

    @staticmethod
    def get_players_result_list_(result):
        """ Return s the result list for each player """
        result_list = []
        for r in result:
            result_list.append(GameResult.get_result_list(r=r))
        return result_list

    @staticmethod
    def get_result_list(r: 'GameResult'):
        """ Return a list indicating the score in a list
            [x, y, z], where x = win, y = lose, and z = draw. """
        x = [0, 0, 0]
        if r == GameResult.WIN:
            x[0] += 1
        elif r == GameResult.LOSE:
            x[1] += 1
        elif r == GameResult.DRAW:
            x[2] += 1
        return x

    @staticmethod
    def get_num_outcomes():
        return 3

    @staticmethod
    def get_new_result_list():
        return [0 for _ in range(GameResult.get_num_outcomes())]
