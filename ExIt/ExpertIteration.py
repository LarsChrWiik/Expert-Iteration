
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from Support.Timer import Timer
import numpy as np
from random import uniform as rnd_float
import random


timer = Timer()


class ExpertIteration:

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = DataSet()
        self.games_played = 0

    def start_ex_it(self, game_class, num_iteration, search_time: float):
        """ Start Expert Iteration to master the given game.
            This process is time consuming. """

        for i in range(num_iteration):

            state = game_class()
            X = state.get_feature_vector(state.turn)

            while not state.is_game_over():
                self.ex_it_state(state=state, search_time=search_time)
            self.games_played += 1

            # TODO: Delete later.
            print("*** Iteration = " + str(self.games_played) + " ***")

            s_array, pi_array, v_array = self.data_set.extract_data()
            self.apprentice.train(X=s_array, Y_pi=pi_array, Y_v=v_array)

            print("v_array = ", v_array)
            print("     pi = ", self.apprentice.pred_prob(X=X))
            print(" pi_new = ", pi_array[0])
            print("v_start = ", self.apprentice.pred_eval(X=X))
            print("")
            print("")
            print("")

    def ex_it_state(self, state: BaseGame, search_time: float):
        """ Expert Iteration for a given state """
        v_values, action_indexes, v = self.expert.search(
            state=state,
            predictor=self.apprentice,
            search_time=search_time
        )
        #pi = self.softmax(v_values=v_values, lower_bound=-1, upper_bound=1)

        action_index = self.e_greedy(v_values=v_values, action_indexes=action_indexes)

        self.data_set.add_sample(state=state, action_index=action_index, v=v)
        state.advance(action_index=action_index)

    def e_greedy(self, v_values, action_indexes):
        e = 0.1
        if rnd_float(0, 1) < e:
            return random.choice(action_indexes)
        return action_indexes[np.argmax(v_values)]


    @staticmethod
    def softmax(v_values, upper_bound, lower_bound):
        """ Softmax function within a certain bound """
        # Restrict to bounds.
        for i, v in enumerate(v_values):
            if v < lower_bound:
                v_values[i] = lower_bound
            elif v > upper_bound:
                v_values[i] = upper_bound

        v_values = [v + abs(lower_bound) for v in v_values]
        sum_ = sum(v_values)
        if sum_ != 0:
            return [v / sum_ for v in v_values]
        return [1.0 / len(v_values) for _ in range(len(v_values))]

    @staticmethod
    def get_action_index_exploit(pi):
        return np.argmax(pi)

    @staticmethod
    def get_action_index_explore(pi, action_indexes):
        """ Choose random action index with respect to action probability.
            Pi must be values between 0 and 1. """
        return np.random.choice(a=action_indexes, size=1, p=pi)[0]
