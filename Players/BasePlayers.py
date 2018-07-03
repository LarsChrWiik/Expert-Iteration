
from Games.GameLogic import BaseGame
from ExIt.ExpertIteration import ExpertIteration
from random import choice as rnd_choice
from ExIt.Evaluator import get_reward_for_action
from ExIt.ActionPolicy import e_greedy, get_action_index_exploit
import numpy as np


def assign_game_index(players):
    # Assign the players a unique index within the game.
    for index, p in enumerate(players):
        p.game_index = index


class BasePlayer:
    """ Base player that is able to move """

    def __init__(self):
        # Unique index of a player.
        self.index = None
        # Unique index within each game. This can change between games.
        self.game_index = None

    def move(self, state: BaseGame, randomness=False):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def move_random(game: BaseGame):
        legal_moves = game.get_legal_moves()
        action_index = rnd_choice(legal_moves)
        game.advance(a=action_index)


class BaseExItPlayer(BasePlayer):
    """ Base player that is able to improve its strategy using Expert Iteration """

    """ Exploration degree that is used in move function.
        This allows game states to differ during comparisons. """
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
        pi = self.ex_it_algorithm.apprentice.pred_prob(fv)
        legal_moves = state.get_legal_moves()

        # Remove PI values that are not legal moves.
        pi_new = [x for i, x in enumerate(pi) if i in legal_moves]
        pi = pi_new

        best_action, action_index = e_greedy(
            pi=pi,
            legal_moves=legal_moves,
            e=BaseExItPlayer.exploration_degree
        )

        if print_info:
            self.print_info(state=state, action_index=action_index)

        if randomness:
            state.advance(a=action_index)
        else:
            state.advance(a=best_action)

    def start_ex_it(self, epochs, search_time):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.ex_it_algorithm.start_ex_it(
            epochs=epochs,
            search_time=search_time
        )

    # TODO: Remove later (Used for testing).
    def print_info(self, state, action_index):
        print("fv = ", state.get_feature_vector())
        print("evaluation = ", self.ex_it_algorithm.apprentice.pred_eval(
            X=state.get_feature_vector()))
        print("action prob = ", self.ex_it_algorithm.apprentice.pred_prob(
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