
from Games.GameLogic import BaseGame
import numpy as np
import random


batch_size = 1024
max_memory_size = 50000


class DataSet:
    """ Class that contains the data set logic for improving the Apprentice """

    def __init__(self):
        self.memory = {}

    def save_samples_in_memory(self, s_array, p_array, v_array):
        """ Stores samples from the previous game and clears the history """
        for i, s in enumerate(s_array):
            self.memory[tuple(s)] = tuple(p_array[i]), v_array[i]
        while len(self.memory) > max_memory_size:
            rnd_index = random.choice(list(self.memory.keys()))
            self.memory.pop(rnd_index)

    def get_sample_batch(self):
        """ Chooses a random batch from the samples stored in memory """
        states = list(self.memory.keys())
        all_examples_indices = len(states)
        mini_batch_indices = np.random.choice(
            all_examples_indices,
            size=min(batch_size, all_examples_indices),
            replace=False
        )

        # Extract samples and generate inputs and targets.
        X_s = np.array(states)[mini_batch_indices]
        Y_p = [list(self.memory[tuple(key)][0]) for key in X_s]
        Y_v = [self.memory[tuple(key)][1] for key in X_s]

        return X_s, Y_p, Y_v
