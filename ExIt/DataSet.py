
from Games.BaseGame import BaseGame
import numpy as np


class DataSet:
    """
    Class that contains the data set logic for improving
    Policy and Evaluation Predictors.
    """

    s_array = []
    pi_array = []
    r_array = []
    turn_array = []

    def add_sample(self, state: BaseGame, pi_update, r):
        """
        Add information to sample arrays.
        A specific index in the arrays refers to the same sample.

        :param state: BaseGame object.
        :param num_actions: int, number of possible actions.
        :param r: float, evaluation of the possition (This will be updated when the game is over).
        """
        turn = state.turn
        fv = state.get_feature_vector(turn)
        pi = pi_update
        #self.__add_sample(fv=fv, pi=pi, r=r, turn=turn)
        for _ in range(state.num_rotations):
            self.__add_sample(fv=fv, pi=pi, r=r, turn=turn)
            # Rotate feature vector and action probabilities.
            fv = state.rotate_fv(fv)
            pi = state.rotate_pi(pi)

    def __add_sample(self, fv, pi, r, turn):
        self.s_array.append(fv)
        self.pi_array.append(pi)
        self.r_array.append(r)
        self.turn_array.append(turn)

    def update_reward_soft(self, final_state: BaseGame):
        """
        This will add reward to the improvement data when the game has finished.
        """
        l = len(self.turn_array)
        for i, t in enumerate(self.turn_array):
            f = final_state.get_reward(t)
            v = self.r_array[i]
            d = f - v
            d = d * ((i+1) / (l+1))
            self.r_array[i] = v + d
            #print("was: " + str(v) + ", r: " + str(f) + ", new: " + str(v + d))

    def update_reward_hard(self, final_state: BaseGame):
        """
        This will add reward to the improvement data when the game has finished.
        """
        for i, t in enumerate(self.turn_array):
            self.r_array[i] = final_state.get_reward(t)

    def extract_data(self):
        """
        Function that extract the data and clear the history.

        :return: array of states, array of action probabilities, array of rewards.
        """
        s_array = self.s_array
        pi_array = self.pi_array
        r_array = self.r_array
        self.clear()
        return s_array, pi_array, r_array

    def clear(self):
        """
        Clear history.
        """
        self.s_array = []
        self.pi_array = []
        self.r_array = []
        self.turn_array = []
