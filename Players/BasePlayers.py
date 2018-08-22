
from Games.GameLogic import BaseGame
from ExIt.ExpertIteration import ExpertIteration
from random import choice as rnd_choice
from ExIt.Evaluator import get_reward_for_action
from ExIt.Policy import e_greedy, exploit_action, explore
import numpy as np
import random


def set_indexes(players: ["BasePlayer"]):
    # Assign the players a unique 'player index' (this index is constant).
    for i, p in enumerate(players):
        p.index = i


class BasePlayer:
    """ Base player that is able to move """

    def __init__(self):
        # Unique index of the player.
        self.index = None

    def move(self, state: BaseGame, verbose=False):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def move_random(state: BaseGame):
        lm = state.get_legal_moves()
        a = rnd_choice(lm)
        state.advance(a)
        return a


class BaseExItPlayer(BasePlayer):
    """ Base player that is able to improve its strategy using Expert Iteration """

    def __init__(self, ex_it_algorithm: ExpertIteration):
        super().__init__()
        self.ex_it_algorithm = ex_it_algorithm
        self.__name__ = ex_it_algorithm.__name__

    def new(self):
        raise NotImplementedError("Please Implement this method")

    def set_game(self, game_class):
        self.ex_it_algorithm.set_game(game_class)

    def set_search_time(self, search_time):
        self.ex_it_algorithm.set_search_time(search_time)

    def move(self, state: BaseGame, verbose=True):
        """ Move according to the apprentice (No expert) """
        fv = state.get_feature_vector()
        pi_pred = self.ex_it_algorithm.apprentice.pred_pi(fv)
        lm = state.get_legal_moves()

        # Remove PI values that are not legal moves.
        pi = [x for i, x in enumerate(pi_pred) if i in lm]
        if verbose:
            print("pi =", pi)

        a = exploit_action(pi, lm)
        state.advance(a)
        return a

    def move_with_search_time(self, state: BaseGame, search_time=None):
        if search_time is not None:
            self.ex_it_algorithm.set_search_time(search_time)
        s, pi, v, t, a = self.ex_it_algorithm.ex_it_state(state)
        state.advance(a)
        return a

    def start_ex_it(self, training_timer, verbose=True):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.ex_it_algorithm.start_ex_it(
            training_timer=training_timer,
            verbose=verbose
        )
