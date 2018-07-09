
from Games.GameLogic import BaseGame
from random import uniform as rnd_float
import numpy as np


class BaseApprentice:
    """ Base class used for policy and evaluation predictors """

    predictor = None

    def init_model(self, input_fv_size, pi_size):
        raise NotImplementedError("Please Implement this method")

    def train(self, X, Y_pi, Y_r):
        raise NotImplementedError("Please Implement this method")

    def pred_v(self, X):
        raise NotImplementedError("Please Implement this method")

    def pred_p(self, X):
        raise NotImplementedError("Please Implement this method")

    def set_model(self, trained_model):
        raise NotImplementedError("Please Implement this method")
