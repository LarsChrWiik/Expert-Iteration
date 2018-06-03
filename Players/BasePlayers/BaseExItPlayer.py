
from Players.BasePlayers.BasePlayer import BasePlayer
from Games.BaseGame import BaseGame
from ExIt.ExpertIteration import ExpertIteration


class BaseExItPlayer(BasePlayer):
    """
    Player that is able to improve its strategy using Expert Iteration.
    """

    __exIt_algorithm = None

    def __init__(self, exIt_algorithm: ExpertIteration):
        self.__exIt_algorithm = exIt_algorithm

    def start_exIt(self, game_class, num_iteration, randomness: bool):
        """
        Starts Expert Iteration to master the given game.
        This process might take some time.
        """
        self.__exIt_algorithm.start_exIt(game_class=game_class, num_iteration=num_iteration,
                                         randomness=randomness)

    def make_expert_move(self, state: BaseGame):
        """
        This is used after the Expert Iteration has completed.

        :param state: BaseGame object.
        """

        print("fv = ", state.get_feature_vector(state.turn))
        print("evaluation = ", self.__exIt_algorithm.apprentice.pred_eval(
            X=state.get_feature_vector(state.turn)))

        print("action prob = ", self.__exIt_algorithm.apprentice.pred_prob(
            X=state.get_feature_vector(state.turn)))
        print("turn = " + str(state.turn))

        action_index, eval, pi_update = self.__exIt_algorithm.calculate_best_action(state)
        print("action_index = " + str(action_index))
        print("policy improver eval = " + str(eval))
        print("updated pi_update = " + str(pi_update))
        state.advance(action_index=action_index)
