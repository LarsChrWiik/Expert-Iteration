
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from random import uniform
from random import choice as rnd_choice
from Support.Timer import Timer
import numpy as np


timer = Timer()


class ExpertIteration:

    @staticmethod
    def get_rnd_prob(iteration, num_iteration):
        return 0.1
        #return 1 - (iteration / num_iteration)

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = DataSet()
        self.games_played = 0

    def apprentice_output(self, state: BaseGame):
        """ Evaluate state and calculates the best action index
            using only the apprentice """
        fv = state.get_feature_vector(state.turn)
        action_indexes = self.apprentice.pred_prob(fv)
        # Remove illegal moves.
        legal_moves = state.get_possible_actions()
        for i, v in enumerate(action_indexes):
            if i not in legal_moves:
                action_indexes[i] = 0
        # Choose best action.
        action_index = np.argmax(action_indexes)
        evaluation = self.apprentice.pred_eval(fv)
        return action_index, evaluation

    def start_ex_it(self, game_class, num_iteration, add_randomness: bool, search_time: float):
        """ Start Expert Iteration to master the given game.
            This process is time consuming. """

        rnd_prob = 0.0
        for i in range(num_iteration):

            state = game_class()
            X = state.get_feature_vector(state.turn)

            # TODO: Change later.
            if add_randomness:
                rnd_prob = ExpertIteration.get_rnd_prob(iteration=i, num_iteration=num_iteration)

            while not state.is_game_over():
                self.ex_it_state(state=state, rnd_prob=rnd_prob, search_time=search_time)
            self.games_played += 1
            #self.data_set.update_reward_hard(state)
            s_array, pi_array, r_array = self.data_set.extract_data()
            self.apprentice.train(X=s_array, Y_pi=pi_array, Y_r=r_array)

            # TODO: Delete later.
            print("*** Iteration = " + str(self.games_played) + " ***")
            print("randomness = " + str(rnd_prob))
            print("r_array = ", self.data_set.r_array)
            print("     pi = ", self.apprentice.pred_prob(X=X))
            print(" pi_new = ", pi_array[0])
            print("r_start = ", self.apprentice.pred_eval(X=X))
            print("")
            print("")
            print("")

    def ex_it_state(self, state: BaseGame, rnd_prob: float, search_time: float):
        """ Expert Iteration for a given state """
        action_index, r = self.expert.search(state=state,
                                             predictor=self.apprentice,
                                             search_time=search_time)
        if uniform(0, 1) < rnd_prob:
            # Random move.
            action_index = rnd_choice(state.get_possible_actions())
            r = self.get_reward_for_action(state=state, action_index=action_index)
        self.data_set.add_sample(state=state, action_index=action_index, r=r)
        state.advance(action_index=action_index)

    def get_reward_for_action(self, state: BaseGame, action_index: int):
        """ Calculates the reward for the given action index """
        c = state.copy()
        c.advance(action_index)
        return self.apprentice.pred_eval(X=c.get_feature_vector(state.turn))
