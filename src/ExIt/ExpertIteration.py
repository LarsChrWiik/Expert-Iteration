
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.DataSet import DataSet
from ExIt.ActionPolicy import explore_action, p_proportional, exploit_action
from ExIt.Expert.Mcts import Mcts
from tqdm import tqdm
import numpy as np
from random import choice as rnd_element
import random


def update_v_values_to_game_outcome(final_state: BaseGame, turn_array, v_array):
    """ Set all v values to the outcome of the game for each player.
        This approach may lead to overfitting. """
    for i, t in enumerate(turn_array):
        v_array[i] = final_state.get_result(t).value
    return v_array


def generate_sample(state: BaseGame, a):
    p_new = np.zeros(state.num_actions, dtype=float)
    p_new[a] = 1
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

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert,
                 use_off_policy=True, data_set=DataSet(), state_branch_degree=0.0):
        self.apprentice = apprentice
        self.expert = expert
        self.data_set = data_set
        self.search_time = None
        self.game_class = None
        self.games_generated = 0
        self.state_branch_degree = state_branch_degree
        self.use_off_policy = use_off_policy
        # Set name.
        self.policy_str = "off-policy" if use_off_policy else "on-policy"
        try:
            self.__name__ = str(type(self.apprentice).__name__) + "_" + str(self.expert.__name__) \
                            + "_" + self.policy_str + "_" + type(self.data_set).__name__ \
                            + "_branch-" + str(state_branch_degree)
        except:
            self.__name__ = str(type(self.apprentice).__name__) + "_" \
                            + str(type(self.expert).__name__) + "_" + self.policy_str \
                            + "_" + type(self.data_set).__name__ \
                            + "_branch-" + str(state_branch_degree)

    def set_game(self, game_class):
        self.game_class = game_class
        self.apprentice.init_model(
            input_fv_size=game_class().fv_size,
            pi_size=game_class().num_actions
        )

    def start_ex_it(self, training_timer, search_time: float, verbose=True):
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
                pi_loss, v_loss = self.apprentice.train(X_s=X_s, Y_p=Y_p, Y_v=Y_v)
                if verbose:
                    return pi_loss, v_loss

        if verbose:
            training_timer.start_new_lap()
            progress_bar = tqdm(range(int(training_timer.version_time)))
            while training_timer.has_time_left():
                pi_loss, v_loss = self_play()
                # Update progress bar.
                progress_bar.update(training_timer.get_time_since_last_check())
                progress_bar.set_postfix(
                    memory_size='%d' % len(self.data_set.memory),
                    games_generated='%d' % self.games_generated,
                    pi_loss='%01.2f' % pi_loss,
                    v_loss='%01.2f' % v_loss
                )
            progress_bar.close()
        else:
            training_timer.start_new_lap()
            while training_timer.has_time_left():
                self_play()

    def ex_it_game(self, state):
        s_array, pi_array, v_array, turn_array = [], [], [], []
        state_copies = []

        while not state.is_game_over():
            s, pi, v, t, a = self.ex_it_state(state)

            if random.uniform(0, 1) < self.state_branch_degree and self.use_off_policy:
                # Make branch from the main line.
                state_copies = add_different_advance(
                    state=state,
                    best_action=a,
                    state_copies=state_copies
                )

            state.advance(a)
            # Store info.
            s_array.append(s)
            pi_array.append(pi)
            v_array.append(v)
            turn_array.append(t)

        # Update the v targets according to the outcome of the game.
        v_array = update_v_values_to_game_outcome(state, turn_array, v_array)

        for s in state_copies:
            s_array2, p_array2, v_array2 = self.ex_it_game(state=s)
            s_array.extend(s_array2)
            pi_array.extend(p_array2)
            v_array.extend(v_array2)

        self.games_generated += 1

        return s_array, pi_array, v_array

    def ex_it_state(self, state: BaseGame):
        """ Expert Iteration for a given state """

        a, a_best, v = self.expert.search(state, self.apprentice, self.search_time, self.use_off_policy)

        s, pi, t = generate_sample(state, a_best)
        return s, pi, v, t, a
