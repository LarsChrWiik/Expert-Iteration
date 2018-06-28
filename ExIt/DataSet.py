
from Games.GameLogic import BaseGame
import numpy as np


batch_size = 256
max_memory_size = 10000


class DataSet:
    """ Class that contains the data set logic for improving the Apprentice """

    def __init__(self):
        self.clear()
        self.memory = None

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

    def save_samples_in_memory(self):
        """ Stores samples from the previous game and clears the history """
        new_data_tuple = [self.s_array, self.pi_array, self.v_array]
        if self.memory is None:
            self.memory = new_data_tuple
        else:
            for i in range(len(self.memory)):
                self.memory[i].extend(new_data_tuple[i])
            self.memory = [t[-max_memory_size:] for t in self.memory]
        self.clear()

    def get_sample_batch(self):
        """ Chooses a random batch from the samples stored in memory """
        sample_size = len(self.memory[0])
        size = min(batch_size, sample_size)
        mini_batch_indices = np.random.choice(
            sample_size,
            size=size,
            replace=False
        )
        return list([np.array(k)[mini_batch_indices] for k in self.memory])

    def clear(self):
        """ Clears the history from previous game """
        self.s_array, self.pi_array, self.v_array, self.turn_array = [[] for _ in range(4)]
