
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.Memory import MemoryList
from ExIt.Policy import Policy
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
    lm = c.get_legal_moves()
    if len(lm) <= 1:
        return state_copies
    lm = [x for x in lm if x != best_action]
    c.advance(rnd_element(lm))
    state_copies.append(c)
    return state_copies


class ExpertIteration:

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert, policy=Policy.OFF,
                 always_exploit=False, memory=None, branch_prob=0.0, growing_search=None):
        self.apprentice = apprentice
        self.expert = expert
        if memory is None:
            self.memory = MemoryList()
        else:
            self.memory = memory
        self.__search_time = None
        self.game_class = None
        self.games_generated = 0
        self.state_branch_degree = branch_prob
        self.policy = policy
        self.always_exploit = always_exploit
        self.growing_search = growing_search
        # Set name.
        extra_name = ""
        if self.apprentice.use_custom_loss:
            extra_name += "_Custom-loss"
        if self.growing_search is not None:
            extra_name += "_Search-grow-" + str(growing_search)
        if not (isinstance(self.memory, MemoryList) and branch_prob == 0.0):
            extra_name += "_" + type(self.memory).__name__ + "_Branch-" + str(branch_prob)
        if always_exploit:
            extra_name += "_Exploit"
        self.__name__ = str(type(self.apprentice).__name__) + "_" + str(self.expert.__name__) \
                        + "_" + str(self.policy.value) + extra_name

    def set_game(self, game_class):
        self.game_class = game_class
        self.apprentice.init_model(
            input_fv_size=game_class().fv_size,
            pi_size=game_class().num_actions
        )

    def set_search_time(self, search_time):
        self.__search_time = search_time

    def start_ex_it(self, training_timer, verbose=True):
        """ Start Expert Iteration to master the given game.
            This process is time consuming. """

        def self_play():
            if self.growing_search is not None:
                self.__search_time += self.growing_search
            self.games_generated = 0
            state = self.game_class()
            s_array, p_array, v_array = self.ex_it_game(state, training_timer)

            # If the time is up, don't train since it will result in an unfair advantage.
            if training_timer.has_time_left():
                # Store game history samples.
                self.memory.save(
                    s_array=s_array,
                    p_array=p_array,
                    v_array=v_array
                )

                # Train on mini-batches.
                for _ in range(self.games_generated):
                    X_s, Y_p, Y_v = self.memory.get_batch()
                    pi_loss, v_loss = self.apprentice.train(X_s=X_s, Y_p=Y_p, Y_v=Y_v)
                    if verbose:
                        return pi_loss, v_loss
            else:
                return None, None

        if verbose:
            training_timer.start_new_lap()
            progress_bar = tqdm(range(int(training_timer.version_time)))
            progress_bar.set_description("Training " + self.__name__)
            while training_timer.has_time_left():
                pi_loss, v_loss = self_play()
                progress_bar.update(training_timer.get_time_since_last_check())
                if not training_timer.has_time_left():
                    break
                # Update progress bar.
                progress_bar.set_postfix(
                    memory_size='%d' % self.memory.get_size(),
                    games_generated='%d' % self.games_generated,
                    pi_loss='%.2f' % pi_loss,
                    v_loss='%.2f' % v_loss,
                    search_time='%.4f' % self.__search_time
                )
            progress_bar.close()
        else:
            training_timer.start_new_lap()
            while training_timer.has_time_left():
                self_play()

    def ex_it_game(self, state, training_timer=None):
        s_array, pi_array, v_array, turn_array = [], [], [], []
        state_copies = []

        while not state.is_game_over():
            if training_timer is not None and not training_timer.has_time_left():
                return None, None, None
            s, pi, v, t, a = self.ex_it_state(state)

            if random.uniform(0, 1) < self.state_branch_degree:
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
        a, a_best, v = self.expert.search(
            state, self.apprentice, self.__search_time, self.always_exploit
        )
        s, pi, t = generate_sample(state, a_best if self.policy == Policy.OFF else a)
        return s, pi, v, t, a
