
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from Support.Timer import Timer
from ExIt.Evaluator import get_reward_for_action
from ExIt.ActionPolicy import e_greedy


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
            X = state.get_feature_vector()

            while not state.is_game_over():
                # TODO: test Spyros theory.
                #Qs = [get_reward_for_action(state, a, self.apprentice) for a in state.get_legal_moves()]
                #print("Qs = ", Qs)
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
        v_values, action_indexes, v = self.expert.search(state=state, predictor=self.apprentice, search_time=search_time)

        action_index = e_greedy(pi=v_values, legal_moves=action_indexes)

        self.data_set.add_sample(state=state, action_index=action_index, v=v)
        state.advance(a=action_index)
