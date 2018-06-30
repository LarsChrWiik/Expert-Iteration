
from Games.GameLogic import BaseGame
import numpy as np


batch_size = 1024
max_memory_size = 10000


class DataSet:
    """ Class that contains the data set logic for improving the Apprentice """

    def __init__(self):
        self.memory = None

    def save_samples_in_memory(self, s_array, pi_array, v_array):
        """ Stores samples from the previous game and clears the history """
        new_data_tuple = [s_array, pi_array, v_array]
        if self.memory is None:
            self.memory = new_data_tuple
        else:
            for i in range(len(self.memory)):
                self.memory[i].extend(new_data_tuple[i])
            self.memory = [t[-max_memory_size:] for t in self.memory]

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

    @staticmethod
    def __blend(a, b):
        blend = []
        for x, y in zip(a, b):
            blend.append((x + y) / 2)
        s = sum(blend)
        return [x / s for x in blend]