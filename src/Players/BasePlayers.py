
from Games.GameLogic import BaseGame
from ExIt.ExpertIteration import ExpertIteration
from random import choice as rnd_choice
from ExIt.Evaluator import get_reward_for_action
from ExIt.ActionPolicy import e_greedy
import numpy as np


def set_indexes(players: ["BasePlayer"]):
    # Assign the players a unique 'player index' (this index is constant).
    for i, p in enumerate(players):
        p.index = i


class BasePlayer:
    """ Base player that is able to move """

    def __init__(self):
        # Unique index of a player.
        self.index = None

    def __name__(self):
        raise NotImplementedError("Please Implement this method")

    def move(self, state: BaseGame, randomness=False):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def move_random(state: BaseGame):
        legal_moves = state.get_legal_moves()
        a = rnd_choice(legal_moves)
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

    def __name__(self):
        return self.ex_it_algorithm.__name__

    def set_game(self, game_class):
        self.ex_it_algorithm.set_game(game_class)

    def move(self, state: BaseGame, randomness=True, print_info=False):
        """ Move according to the apprentice (No expert) """
        fv = state.get_feature_vector()
        p_pred = self.ex_it_algorithm.apprentice.pred_p(fv)
        v_pred = self.ex_it_algorithm.apprentice.pred_v(fv)
        legal_moves = state.get_legal_moves()

        # Remove PI values that are not legal moves.
        p = [x for i, x in enumerate(p_pred) if i in legal_moves]

        best_action, action_index = e_greedy(
            p=p,
            legal_moves=legal_moves,
            e=BaseExItPlayer.exploration_degree
        )

        if print_info:
            self.print_info(state=state, action_index=action_index)

        a = action_index if randomness else best_action
        state.advance(a)
        return a, p_pred, v_pred

    def start_ex_it(self, epochs, search_time):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.ex_it_algorithm.start_ex_it(
            epochs=epochs,
            search_time=search_time
        )

    # TODO: Remove later (Used for testing).
    def print_info(self, state, action_index):
        print("fv = ", state.get_feature_vector())
        print("evaluation = ", self.ex_it_algorithm.apprentice.pred_v(
            X=state.get_feature_vector()))
        print("action prob = ", self.ex_it_algorithm.apprentice.pred_p(
            X=state.get_feature_vector()))
        print("action_index = " + str(action_index))
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1
        print("updated pi_update = " + str(pi_update))

    # TODO: Remove later. (Used for testing).
    def print_Q_info(self, state, pi, legal_moves):
        Qs = [get_reward_for_action(state, a, self.ex_it_algorithm.apprentice) for a in state.get_legal_moves()]
        v_values, action_indexes, v = self.ex_it_algorithm.expert.search(
            state=state,
            predictor=self.ex_it_algorithm.apprentice,
            search_time=10
        )
        print("pi = ", pi)
        print("legal_moves = ", legal_moves)
        print("Qs = ", Qs)
        print("v_values = ", v_values)
        print("action_indexes = ", action_indexes)
        print("")
