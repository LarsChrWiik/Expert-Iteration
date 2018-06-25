
from Games.GameLogic import BaseGame
import numpy as np


batch_size = 128
max_experience_replay_memory_size = 5000


class DataSet:
    """ Class that contains the data set logic for improving the Apprentice """

    def __init__(self):
        self.__clear()
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

    def extract_data(self):
        """ Function that extract the data and clear the history """
        new_data_tuple = [self.s_array, self.pi_array, self.v_array]
        if self.memory is None:
            self.memory = new_data_tuple
        else:
            for i in range(len(self.memory)):
                self.memory[i].extend(new_data_tuple[i])
            self.memory = [t[-max_experience_replay_memory_size:] for t in self.memory]

        self.__clear()
        all_examples_indices = len(self.memory[0])
        mini_batch_indices = np.random.choice(
            all_examples_indices,
            size=min(batch_size, all_examples_indices),
            replace=False
        )

        return list([np.array(k)[mini_batch_indices] for k in self.memory])

    def __clear(self):
        self.s_array, self.pi_array, self.v_array, self.turn_array = [[] for _ in range(4)]
