
from ExIt.Apprentice.BaseApprentice import BaseApprentice
import random


class RandomPredictor(BaseApprentice):
    """ This class is used for making random predictions """

    def init_model(self, input_fv_size, pi_size):
        self.input_fv_size = input_fv_size
        self.pi_size = pi_size

    def pred_v(self, X):
        return random.uniform(-0.5, 0.5)

    def pred_pi(self, X):
        pi = [random.uniform(0, 1) for _ in range(len(X))]
        s = sum(pi)
        return [p / s for p in pi]

    def train(self, X, Y_pi, Y_r):
        raise NotImplementedError("The Random Predictor does not have this functionality. ")

    def set_model(self, trained_model):
        raise NotImplementedError("The Random Predictor does not have this functionality. ")
