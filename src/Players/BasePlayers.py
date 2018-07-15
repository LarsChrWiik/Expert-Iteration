
from Games.GameLogic import BaseGame
from ExIt.ExpertIteration import ExpertIteration
from random import choice as rnd_choice
from ExIt.Evaluator import get_reward_for_action
from ExIt.Policy import e_greedy, exploit_action
import numpy as np


def set_indexes(players: ["BasePlayer"]):
    # Assign the players a unique 'player index' (this index is constant).
    for i, p in enumerate(players):
        p.index = i


class BasePlayer:
    """ Base player that is able to move """

    def __init__(self):
        # Unique index of the player.
        self.index = None

    def move(self, state: BaseGame, randomness=False):
        raise NotImplementedError("Please Implement this method")

    def new_player(self):
        """ Returns a new player with the same init parameters """
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def move_random(state: BaseGame):
        lm = state.get_legal_moves()
        a = rnd_choice(lm)
        state.advance(a)
        return a


class BaseExItPlayer(BasePlayer):
    """ Base player that is able to improve its strategy using Expert Iteration """

    """ Exploration degree that is used in move function.
        This allows some randomness when comparing two ExIt-algorithms. """
    exploration_degree = 0.1

    def __init__(self, ex_it_algorithm: ExpertIteration):
        super().__init__()
        self.ex_it_algorithm = ex_it_algorithm
        self.__name__ = ex_it_algorithm.__name__

    def set_game(self, game_class):
        self.ex_it_algorithm.set_game(game_class)

    def new_player(self):
        raise NotImplementedError("Please Implement this method")

    def move(self, state: BaseGame, randomness=True, print_info=False):
        """ Move according to the apprentice (No expert) """
        fv = state.get_feature_vector()
        p_pred = self.ex_it_algorithm.apprentice.pred_pi(fv)
        v_pred = self.ex_it_algorithm.apprentice.pred_v(fv)
        lm = state.get_legal_moves()

        # Remove PI values that are not legal moves.
        pi = [x for i, x in enumerate(p_pred) if i in lm]

        action_index = e_greedy(
            xi=pi,
            lm=lm,
            e=BaseExItPlayer.exploration_degree
        )
        best_action = exploit_action(pi, lm)

        a = action_index if randomness else best_action
        state.advance(a)
        return a, p_pred, v_pred

    def start_ex_it(self, training_timer, search_time):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.ex_it_algorithm.start_ex_it(
            training_timer=training_timer,
            search_time=search_time
        )
