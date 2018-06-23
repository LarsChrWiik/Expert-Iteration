
from Games.GameLogic import BaseGame
import numpy as np


class DataSet:
    """ Class that contains the data set logic for improving the Apprentice """

    def __init__(self):
        self.__clear()

    def add_sample(self, state: BaseGame, action_index, v):
        """ Add information to sample arrays. Array indexes refer to the same sample """
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1
        turn, fv, pi = state.turn, state.get_feature_vector(), pi_update
        self.s_array.append(fv)
        self.pi_array.append(pi)
        self.v_array.append(v)
        self.turn_array.append(turn)

    def set_game_outcome_v_values(self, final_state: BaseGame):
        """ Set all v values to the outcome of the game for each player.
            This approach may lead to overfitting. """
        for i, t in enumerate(self.turn_array):
            self.v_array[i] = final_state.get_result(t).value

    def extract_data(self):
        """ Function that extract the data and clear the history """
        data_tuple = self.s_array, self.pi_array, self.v_array
        self.__clear()
        return data_tuple

    def __clear(self):
        self.s_array, self.pi_array, self.v_array, self.turn_array = [[] for _ in range(4)]
