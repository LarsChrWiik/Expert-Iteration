
from Games.GameLogic import BaseGame
import numpy as np


class DataSet:
    """ Class that contains the data set logic for improving the Apprentice """

    def __init__(self):
        self.__clear()

    def add_sample(self, state: BaseGame, action_index, r):
        """ Add information to sample arrays. Array indexes refer to the same sample. """
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1
        turn, fv, pi = state.turn, state.get_feature_vector(state.turn), pi_update
        self.__add_sample(fv=fv, pi=pi, r=r, turn=turn)

    def extract_data(self):
        """ Function that extract the data and clear the history """
        data_tuple = self.s_array, self.pi_array, self.r_array
        self.__clear()
        return data_tuple

    # TODO: Not used. (Remove later).
    def update_reward_soft(self, final_state: BaseGame):
        """ Add relative reward to the improvement data when the game has finished """
        play_len = len(self.turn_array)
        for i, t in enumerate(self.turn_array):
            f = final_state.get_reward(t)
            v = self.r_array[i]
            d = f - v
            d = d * ((i+1) / (play_len+1))
            self.r_array[i] = v + d

    def update_reward_hard(self, final_state: BaseGame):
        """ Set reward to the improvement data when the game has finished """
        for i, t in enumerate(self.turn_array):
            self.r_array[i] = final_state.get_reward(t)

    def __add_sample(self, fv, pi, r, turn):
        self.s_array.append(fv)
        self.pi_array.append(pi)
        self.r_array.append(r)
        self.turn_array.append(turn)

    def __clear(self):
        self.s_array, self.pi_array, self.r_array, self.turn_array = [[] for _ in range(4)]
