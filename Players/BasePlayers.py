
from Games.GameLogic import BaseGame
from ExIt.ExpertIteration import ExpertIteration
from random import choice as rnd_choice
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
        # Name of the class of the game.
        self.game_class = None

    def set_game(self, game_class):
        self.game_class = game_class

    def move(self, state: BaseGame):
        raise NotImplementedError("Please Implement this method")

    @staticmethod
    def move_random(game: BaseGame):
        legal_moves = game.get_possible_actions()
        action_index = rnd_choice(legal_moves)
        game.advance(action_index=action_index)


class BaseExItPlayer(BasePlayer):
    """ Base player that is able to improve its strategy using Expert Iteration """

    def __init__(self, ex_it_algorithm: ExpertIteration):
        super().__init__()
        self.__ex_it_algorithm = ex_it_algorithm

    def set_game(self, game_class):
        self.game_class = game_class
        self.__ex_it_algorithm.apprentice.init_model(input_fv_size=game_class().fv_size,
                                                     pi_size=game_class().num_actions)

    def move(self, state: BaseGame, randomness=True, print_info=False):
        """ Move according to the apprentice (No expert) """
        fv = state.get_feature_vector(state.turn)
        pi = self.__ex_it_algorithm.apprentice.pred_prob(fv)
        legal_moves = state.get_possible_actions()
        if not randomness:
            for i, x in enumerate(pi):
                if i not in legal_moves:
                    pi[i] = 0
            action_index = self.__ex_it_algorithm.get_action_index_exploit(pi=pi)
        else:
            action_index = self.__ex_it_algorithm.get_action_index_explore(
                pi=pi,
                action_indexes=legal_moves
            )

        if print_info:
            self.print_info(state=state, action_index=action_index)
        state.advance(action_index=action_index)

    def start_ex_it(self, num_iteration, randomness: bool, search_time: float):
        """ Starts Expert Iteration. NB: Time consuming process """
        self.__ex_it_algorithm.start_ex_it(
            game_class=self.game_class,
            num_iteration=num_iteration,
            randomness=randomness,
            search_time=search_time
        )

    # TODO: Remove later (Used for testing).
    def print_info(self, state, action_index):
        print("fv = ", state.get_feature_vector(state.turn))
        print("evaluation = ", self.__ex_it_algorithm.apprentice.pred_eval(
            X=state.get_feature_vector(state.turn)))
        print("action prob = ", self.__ex_it_algorithm.apprentice.pred_prob(
            X=state.get_feature_vector(state.turn)))
        print("action_index = " + str(action_index))
        pi_update = np.zeros(state.num_actions, dtype=float)
        pi_update[action_index] = 1
        print("updated pi_update = " + str(pi_update))
