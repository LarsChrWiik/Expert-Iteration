

from ExIt.ExpertIteration import ExpertIteration
from Games.BaseGame import BaseGame
import numpy as np


class BasePlayer:
    """ Base player that is able to move """

    def __init__(self):
        self.player_index = None

    def move(self, state: BaseGame):
        raise NotImplementedError("Please Implement this method")


class BaseExItPlayer(BasePlayer):
    """ Base player that is able to improve its strategy using Expert Iteration """

    def __init__(self, ex_it_algorithm: ExpertIteration):
        super().__init__()
        self.__ex_it_algorithm = ex_it_algorithm

    def start_ex_it(self, game_class, num_iteration, randomness: bool):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.__ex_it_algorithm.start_ex_it(game_class=game_class, num_iteration=num_iteration,
                                           add_randomness=randomness)

    def move(self, state: BaseGame):
        """ Calculate the best move """
        action_index, evaluation = self.__ex_it_algorithm.calculate_best_action(state)

        # TODO: Remove later (Used for testing).
        print("fv = ", state.get_feature_vector(state.turn))
        print("evaluation = ", self.__ex_it_algorithm.apprentice.pred_eval(
            X=state.get_feature_vector(state.turn)))
        print("action prob = ", self.__ex_it_algorithm.apprentice.pred_prob(
            X=state.get_feature_vector(state.turn)))
        print("turn = " + str(state.turn))
        print("action_index = " + str(action_index))
        print("policy improver eval = " + str(evaluation))
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1
        print("updated pi_update = " + str(pi_update))

        state.advance(action_index=action_index)
