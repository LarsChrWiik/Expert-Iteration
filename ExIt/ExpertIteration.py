
from ExIt.Apprentice import BaseApprentice
from ExIt.Expert import BaseExpert
from Games.GameLogic import BaseGame
from ExIt.Memory import MemoryList, MemorySet, MemoryListGrowing, MemorySetAvg
from ExIt.Policy import Policy
from tqdm import tqdm
import numpy as np
from random import choice as rnd_element
import random


DEFAULT_GROWING_SEARCH_VALUE = 0.0001
DEFAULT_MIN_SEARCH_TIME = 0.05


def get_growing_search_val(growing_search_time):
    """ Return the growing search time value given kwargs bool """
    if isinstance(growing_search_time, bool):
        return DEFAULT_GROWING_SEARCH_VALUE if growing_search_time else None
    return growing_search_time


def update_v_values_to_game_outcome(final_state: BaseGame, turn_array, v_array):
    """ Set all v values to the outcome of the game for each player.
        This approach may lead to overfitting. """
    for i, t in enumerate(turn_array):
        v_array[i] = final_state.get_result(t).value
    return v_array


def generate_pi(state: BaseGame, a):
    """ Generate action probability pi given an action index """
    p_new = np.zeros(state.num_actions, dtype=float)
    p_new[a] = 1
    return p_new


def get_branch_state(state, best_action):
    """ Return a branch state """
    c = state.copy()
    lm = c.get_legal_moves()
    if len(lm) <= 1:
        return None
    lm = [x for x in lm if x != best_action]
    c.advance(rnd_element(lm))
    return c


def update_existing(main_kwargs, kwargs):
    for key, value in kwargs.items():
        if key in main_kwargs:
            main_kwargs[key] = value
    return main_kwargs


class ExpertIteration:

    default_kwargs = {
        "policy": Policy.OFF,
        "growing_search_time": False,
        "growing_depth": False,
        "min_growing_time": None,
        "soft_z": False,
        "always_exploit": False,
        "memory": "default",
        "branch_prob": 0.0
    }

    def __init__(self, apprentice: BaseApprentice, expert: BaseExpert, **kwargs):
        self.kwargs = self.default_kwargs.copy()
        self.kwargs = update_existing(self.kwargs, kwargs)

        self.apprentice = apprentice
        self.expert = expert

        # ***** Init parameters *****
        self.game_class = None
        self.games_generated = 0
        self.soft_z = self.kwargs.get("soft_z")
        self.policy = self.kwargs.get("policy")
        self.state_branch_degree = self.kwargs.get("branch_prob")
        self.use_growing_search_time = self.kwargs.get("growing_search_time")
        self.growing_search_val = get_growing_search_val(self.kwargs.get("growing_search_time"))
        self.__search_time = DEFAULT_MIN_SEARCH_TIME if self.growing_search_val is not None else None

        # ***** Calculate an appropriate name for this expert iteration variant *****
        extra_name = ""
        if self.kwargs.get("memory") in [None, "default", "MemoryList"]:
            self.memory = MemoryList()
        elif self.kwargs.get("memory") == "MemorySet":
            self.memory = MemorySet()
            extra_name += "_MemSet"
        elif self.kwargs.get("memory") == "MemorySetAvg":
            self.memory = MemorySetAvg()
            extra_name += "_MemSetAvg"
        elif self.kwargs.get("memory") == "MemoryListGrowing":
            self.memory = MemoryListGrowing()
            extra_name += "_MemGrow"
        else:
            raise Exception("Unknown Memory object!")

        if self.kwargs.get("growing_depth"):
            extra_name += "_Grow-depth"
            self.expert.fixed_depth = 1
        if self.soft_z:
            extra_name += "_Soft-Z"
        if self.apprentice.use_custom_loss:
            extra_name += "_Custom-loss"
        if self.use_growing_search_time:
            min_time = DEFAULT_MIN_SEARCH_TIME if self.kwargs.get("min_growing_time") is None \
                else self.kwargs.get("min_growing_time")
            extra_name += "_Grow-" + str(min_time) + "+" + str(self.growing_search_val)
        if self.state_branch_degree > 0.0:
            extra_name += "_Branch-" + str(self.state_branch_degree)
        if self.kwargs.get("always_exploit"):
            extra_name += "_Exploit"

        # Set same of expert iteration variant.
        self.__name__ = "ExIt_" + str(type(self.apprentice).__name__) + "_" \
                        + str(self.expert.__name__) + "_" + str(self.policy.value) + extra_name

    def set_game(self, game_class):
        self.game_class = game_class.new()
        self.apprentice.init_model(
            input_fv_size=game_class.fv_size,
            pi_size=game_class.num_actions
        )

    def set_search_time(self, search_time):
        self.__search_time = search_time

    def start_ex_it(self, training_timer, verbose=True):
        """ Start Expert Iteration to master the given game.
            This process is time consuming. """

        def self_play():
            if self.use_growing_search_time:
                self.__search_time += self.growing_search_val
            if self.kwargs.get("growing_depth"):
                if self.memory.should_increment():
                    # This is used for Minimax variants only.
                    self.expert.fixed_depth += 1
            self.games_generated = 0
            state = self.game_class.new()
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
                    mem_size='%d' % self.memory.get_size(),
                    pi_loss='%.2f' % pi_loss,
                    v_loss='%.2f' % v_loss,
                    time='%.4f' % self.__search_time
                )
            progress_bar.close()
        else:
            training_timer.start_new_lap()
            while training_timer.has_time_left():
                self_play()

    def ex_it_game(self, state, training_timer=None):
        """ Self-play a game until it finishes """
        s_array, pi_array, v_array, turn_array = [], [], [], []
        state_copies = []

        while not state.is_game_over():
            if training_timer is not None and not training_timer.has_time_left():
                return None, None, None
            s, pi, v, t, a = self.ex_it_state(state)

            if random.uniform(0, 1) < self.state_branch_degree:
                # Make branch from the main line.
                c_branch = get_branch_state(state=state, best_action=a)
                if c_branch is not None:
                    state_copies.append(c_branch)

            state.advance(a)
            # Store info.
            s_array.append(s)
            pi_array.append(pi)
            v_array.append(v)
            turn_array.append(t)

        if not self.soft_z:
            # Update the v targets according to the outcome of the game.
            v_array = update_v_values_to_game_outcome(state, turn_array, v_array)

        for s in state_copies:
            s_array2, p_array2, v_array2 = self.ex_it_game(state=s)
            s_array.extend(s_array2)
            pi_array.extend(p_array2)
            v_array.extend(v_array2)

        self.games_generated += 1

        s_array, pi_array, v_array = state.add_augmentations(s_array, pi_array, v_array)
        return s_array, pi_array, v_array

    def ex_it_state(self, state: BaseGame):
        """ Expert Iteration for a given state """
        a, a_best, v = self.expert.search(
            state, self.apprentice, self.__search_time, self.kwargs.get("always_exploit")
        )
        if self.policy == Policy.OFF:
            # Uses the optimal action to generate pi target.
            pi = generate_pi(state, a_best)
        else:
            # Uses the exploration action to generate pi target.
            pi = generate_pi(state, a)
        s = state.get_feature_vector()
        t = state.turn
        return s, pi, v, t, a
