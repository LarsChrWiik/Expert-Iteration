
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from Support.Timer import Timer
from ExIt.Evaluator import get_reward_for_action
from ExIt.ActionPolicy import e_greedy
from tqdm import tqdm, trange
from random import choice as rnd_element
import numpy as np


timer = Timer()
# 1.0 = always explore. 0.0 = always exploit.
exploration_degree = 0.1


def set_game_outcome_v_values(final_state: BaseGame, turn_array, v_array):
    """ Set all v values to the outcome of the game for each player.
        This approach may lead to overfitting. """
    for i, t in enumerate(turn_array):
        v_array[i] = final_state.get_result(t).value
    return v_array


def generate_sample(state: BaseGame, action_index):
    pi_new = np.zeros(state.num_actions, dtype=float)
    pi_new[action_index] = 1
    return state.get_feature_vector(), pi_new, state.turn


def add_different_advance(state, action_index, state_copies):
    c = state.copy()
    legal_moves = c.get_legal_moves()
    if len(legal_moves) == 1:
        return state_copies
    legal_moves = [x for x in legal_moves if x != action_index]
    c.advance(rnd_element(legal_moves))
    state_copies.append(c)
    return state_copies


class ExpertIteration:

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = DataSet()
        self.search_time = None

    def start_ex_it(self, game_class, num_iteration, search_time):
        """ Start Expert Iteration to master the given game.
            This process is time consuming. """
        self.search_time = search_time

        with trange(num_iteration) as t:
            for _ in t:
                state = game_class()
                s_array, pi_array, v_array = self.ex_it_game(state=state)

                # Store game history samples.
                self.data_set.save_samples_in_memory(
                    s_array=s_array,
                    pi_array=pi_array,
                    v_array=v_array
                )

                # Train apprentice using on mini-batches.
                s_array, pi_array, v_array = self.data_set.get_sample_batch()
                p, v = self.apprentice.train(X=s_array, Y_pi=pi_array, Y_v=v_array)
                t.set_postfix(pl='%01.2f' % p, vl='%01.2f' % v)

    def ex_it_game(self, state):
        s_array, pi_array, v_array, turn_array = [], [], [], []
        state_copies = []

        while not state.is_game_over():
            s, p, v, t, action_index = self.ex_it_state(state)
            """
            state_copies = add_different_advance(
                state=state,
                action_index=action_index,
                state_copies=state_copies
            )
            """
            state.advance(a=action_index)
            # Store info.
            s_array.append(s)
            pi_array.append(p)
            v_array.append(v)
            turn_array.append(t)

        v_array = set_game_outcome_v_values(
            final_state=state,
            turn_array=turn_array,
            v_array=v_array
        )

        # Add another advance from this state.
        """
        for s in state_copies:
            s_array2, pi_array2, v_array2 = self.ex_it_game(state=s)
            s_array.extend(s_array2)
            pi_array.extend(pi_array2)
            v_array.extend(v_array2)
        """

        return s_array, pi_array, v_array

    def ex_it_state(self, state: BaseGame):
        """ Expert Iteration for a given state.
            s = state, p = action probability, v = value, t = turn. """
        v_values, action_indexes, v = self.expert.search(
            state=state,
            predictor=self.apprentice,
            search_time=self.search_time
        )

        best_action, action_index = e_greedy(
            pi=v_values,
            legal_moves=action_indexes,
            e=exploration_degree
        )

        s, p, t = generate_sample(state=state, action_index=best_action)
        return s, p, v, t, action_index


"""
pi = self.apprentice.pred_prob(X=state.get_feature_vector())
legal_moves = state.get_legal_moves()
for i in range(len(pi)):
    if i not in legal_moves:
        pi[i] = 0
"""


"""
c = state.copy()
    legal_moves = c.get_legal_moves()
    c.advance(rnd_element(legal_moves))
    if not c.is_game_over():
        s_array, pi_array, v_array = self.ex_it_game(state=c)
"""