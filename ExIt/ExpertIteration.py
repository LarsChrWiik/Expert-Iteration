
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from Support.Timer import Timer
from ExIt.Evaluator import get_reward_for_action
from ExIt.ActionPolicy import e_greedy


timer = Timer()
# 1.0 = always explore. 0.0 = always exploit.
exploration_degree = 0.1


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
            X = state.get_feature_vector()

            num_states = 0
            while not state.is_game_over():
                self.ex_it_state(state=state, search_time=search_time)
                num_states += 1
            self.games_played += 1

            self.data_set.set_game_outcome_v_values(final_state=state)

            # TODO: Delete later.
            print("*** Iteration = " + str(self.games_played) + " ***")

            print("v_array = ", self.data_set.v_array[-num_states:])
            print("     pi = ", self.apprentice.pred_prob(X=X))
            print(" pi_new = ", self.data_set.pi_array[-num_states])
            print("v_start = ", self.apprentice.pred_eval(X=X))
            print("")
            print("")
            print("")

        self.data_set.save_samples_in_memory()
        s_array, pi_array, v_array = self.data_set.get_sample_batch()
        self.apprentice.train(X=s_array, Y_pi=pi_array, Y_v=v_array)

    def ex_it_state(self, state: BaseGame, search_time: float):
        """ Expert Iteration for a given state """
        v_values, action_indexes, v = self.expert.search(state=state, predictor=self.apprentice, search_time=search_time)

        best_action, action_index = e_greedy(
            pi=v_values,
            legal_moves=action_indexes,
            e=exploration_degree
        )

        self.data_set.add_sample(state=state, action_index=best_action, v=v)
        state.advance(a=action_index)
