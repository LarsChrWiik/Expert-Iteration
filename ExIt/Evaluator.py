
from Games.GameLogic import BaseGame
from ExIt.Apprentice import BaseApprentice


def zero_sum_2v2_evaluation(state: BaseGame, original_turn: int, predictor: BaseApprentice):
    if state.is_game_over():
        return state.get_result(original_turn).value
    else:
        return predictor.pred_eval(
            X=state.get_feature_vector()
        ) * (1 if state.turn == original_turn else -1)


def get_reward_for_action(state: BaseGame, action_index: int, predictor: BaseApprentice):
    """ Calculates the reward for the given action index.
        This function assumes a zero sum 2v2 game. """
    c = state.copy()
    c.advance(action_index)
    return zero_sum_2v2_evaluation(
        state=c,
        original_turn=state.turn,
        predictor=predictor
    )
