
from Games.BaseGame import BaseGame
from ExIt.ExpertIteration import ExpertIteration
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
        self.game_class = None

    def set_game(self, game_class):
        self.game_class = game_class
        self.__ex_it_algorithm.apprentice.init_model(input_fv_size=game_class().fv_size,
                                                     pi_size=game_class().num_actions)

    def start_ex_it(self, num_iteration, randomness: bool, search_time: float):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.__ex_it_algorithm.start_ex_it(game_class=self.game_class,
                                           num_iteration=num_iteration,
                                           add_randomness=randomness,
                                           search_time=search_time)

    def move(self, state: BaseGame):
        """ Calculate the best move """
        action_index, evaluation = self.__ex_it_algorithm.apprentice_output(state)
        self.print_info(state=state, action_index=action_index)
        state.advance(action_index=action_index)

    # TODO: Remove later (Used for testing).
    def print_info(self, state, action_index):
        print("fv = ", state.get_feature_vector(state.turn))
        print("evaluation = ", self.__ex_it_algorithm.apprentice.pred_eval(
            X=state.get_feature_vector(state.turn)))
        print("action prob = ", self.__ex_it_algorithm.apprentice.pred_prob(
            X=state.get_feature_vector(state.turn)))
        print("action_index = " + str(action_index))
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1
        print("updated pi_update = " + str(pi_update))
