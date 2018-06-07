
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.BaseGame import BaseGame
from ExIt.DataSet import DataSet
from random import uniform
from random import choice as rnd_choice
import numpy as np


class ExpertIteration:

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = DataSet()

    def calculate_best_action(self, state: BaseGame):
        return self.expert.start(state=state, predictor=self.apprentice)

    def start_ex_it(self, game_class, num_iteration, add_randomness: bool):
        """ Start Expert Iteration to master the given game.
            This process is time consuming """

        self.apprentice.init_model(input_fv_size=game_class().fv_size,
                                   pi_size=game_class().num_actions)

        #changes = [y * 0.1 for y in range(1, 11)]
        #lr_change = [int(num_iteration) * x for x in changes]
        for i in range(num_iteration):
            print("lr = ", float(self.apprentice.get_lr()))
            # TODO: Move to NN maybe?
            #if i in lr_change:
                #self.apprentice.set_lr(new_lr=self.apprentice.get_lr()*0.9)

            state = game_class()

            max_rnd = 1.0
            rnd_prob = 0 if not add_randomness else (1 - (i / num_iteration)) * max_rnd

            X = state.get_feature_vector(state.turn)
            while not state.has_finished():
                self.ex_it_state(state=state, rnd_prob=rnd_prob)
            print("*** Iteration = " + str(i) + " ***")
            print("randomness = " + str(rnd_prob))
            print("    r = ", self.data_set.r_array)
            # TODO: Check this.
            self.data_set.update_reward_hard(state)
            print("r_new = ", self.data_set.r_array)

            # Train the Policy and Evaluation Predictor.
            s_array, pi_array, r_array = self.data_set.extract_data()
            if len(s_array) > 0:
                print("    pi = ", self.apprentice.pred_prob(X=X))
                print("pi_new = ", pi_array[0])
                print("r_start= ", self.apprentice.pred_eval(X=X))
                print("")
                self.apprentice.train(X=s_array, Y_pi=pi_array, Y_r=r_array)
            print("")
            print("")

    def ex_it_state(self, state: BaseGame, rnd_prob: float):
        """ Expert Iteration for a given state """
        action_index, r = self.expert.start(state=state, predictor=self.apprentice)
        if uniform(0, 1) < rnd_prob:
            action_index = rnd_choice(state.get_legal_moves(state.turn))

        # Create action probability.
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1

        self.data_set.add_sample(state=state, pi_update=pi_update, r=r)

        state.advance(action_index=action_index)
