
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from Support.Timer import Timer
from ExIt.Evaluator import get_reward_for_action
from ExIt.ActionPolicy import e_greedy
from tqdm import tqdm, trange


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

        with trange(num_iteration) as t:
            for _ in t:

                state = game_class()
                num_states = 0
                while not state.is_game_over():
                    self.ex_it_state(state=state, search_time=search_time)
                    num_states += 1
                self.games_played += 1

                self.data_set.set_game_outcome_v_values(final_state=state)

                # Store game history samples and train the Apprentice.
                self.data_set.save_samples_in_memory()
                # Train on mini-batches.
                s_array, pi_array, v_array = self.data_set.get_sample_batch()
                p, v = self.apprentice.train(X=s_array, Y_pi=pi_array, Y_v=v_array)
                t.set_postfix(pl='%01.2f' % p, vl='%01.2f' % v)

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
