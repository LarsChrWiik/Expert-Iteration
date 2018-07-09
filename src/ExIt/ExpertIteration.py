
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from Misc.Timer import Timer
from ExIt.ActionPolicy import get_action_index_exploit
from tqdm import trange
import numpy as np
from random import choice as rnd_element
import random


timer = Timer()

# 1.0 = explore. 0.0 = exploit.
exploration_degree = 0.1


def update_v_values_to_game_outcome(final_state: BaseGame, turn_array, v_array):
    """ Set all v values to the outcome of the game for each player.
        This approach may lead to overfitting. """
    for i, t in enumerate(turn_array):
        v_array[i] = final_state.get_result(t).value
    return v_array


def generate_sample(state: BaseGame, action_index):
    p_new = np.zeros(state.num_actions, dtype=float)
    p_new[action_index] = 1
    return state.get_feature_vector(), p_new, state.turn


def add_different_advance(state, best_action, state_copies):
    c = state.copy()
    legal_moves = c.get_legal_moves()
    if len(legal_moves) <= 1:
        return state_copies
    legal_moves = [x for x in legal_moves if x != best_action]
    c.advance(rnd_element(legal_moves))
    state_copies.append(c)
    return state_copies


class ExpertIteration:

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = DataSet()
        self.search_time = None
        self.game_class = None
        self.games_generated = 0
        # Set name.
        try:
            self.__name__ = str(type(self.apprentice).__name__) + "_" + str(self.expert.__name__)
        except:
            self.__name__ = str(type(self.apprentice).__name__) + "_" + str(type(self.expert).__name__)

    def set_game(self, game_class):
        self.game_class = game_class
        self.apprentice.init_model(
            input_fv_size=game_class().fv_size,
            pi_size=game_class().num_actions
        )

    def start_ex_it(self, epochs, search_time: float, verbose=True):
        """ Start Expert Iteration to master the given game.
            This process is time consuming. """
        self.search_time = search_time

        def self_play():
            self.games_generated = 0
            state = self.game_class()
            s_array, p_array, v_array = self.ex_it_game(state)

            # Store game history samples.
            self.data_set.save_samples_in_memory(
                s_array=s_array,
                p_array=p_array,
                v_array=v_array
            )

            # Train on mini-batches.
            for _ in range(self.games_generated):
                X_s, Y_p, Y_v = self.data_set.get_sample_batch()
                p, v = self.apprentice.train(X_s=X_s, Y_p=Y_p, Y_v=Y_v)
                if verbose:
                    return p, v

        if verbose:
            with trange(epochs) as t:
                for _ in t:
                    p, v = self_play()
                    t.set_postfix(
                        memory_size='%d' % len(self.data_set.memory),
                        games_generated='%d' % self.games_generated,
                        pl='%01.2f' % p,
                        vl='%01.2f' % v
                    )
        else:
            for _ in range(epochs):
                self_play()

    def ex_it_game(self, state):
        s_array, p_array, v_array, turn_array = [], [], [], []
        state_copies = []

        while not state.is_game_over():
            s, p, v, t, best_action = self.ex_it_state(state)

            if random.uniform(0, 1) < exploration_degree:
                # Make branch from the main line.
                state_copies = add_different_advance(
                    state=state,
                    best_action=best_action,
                    state_copies=state_copies
                )

            state.advance(a=best_action)
            # Store info.
            s_array.append(s)
            p_array.append(p)
            v_array.append(v)
            turn_array.append(t)

        # Update the v targets according to the outcome of the game.
        v_array = update_v_values_to_game_outcome(state, turn_array, v_array)

        for s in state_copies:
            s_array2, p_array2, v_array2 = self.ex_it_game(state=s)
            s_array.extend(s_array2)
            p_array.extend(p_array2)
            v_array.extend(v_array2)

        self.games_generated += 1

        return s_array, p_array, v_array

    def ex_it_state(self, state: BaseGame):
        """ Expert Iteration for a given state """
        v_values, legal_moves, v = self.expert.search(
            state=state,
            predictor=self.apprentice,
            search_time=self.search_time
        )

        best_action = get_action_index_exploit(values=v_values, legal_moves=legal_moves)

        s, p, t = generate_sample(state=state, action_index=best_action)
        return s, p, v, t, best_action
