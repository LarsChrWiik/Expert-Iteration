
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Support.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation


def alpha_beta_search(state, alpha, beta, depth, predictor, original_turn, timer, is_root):

    def max_value(state, alpha, beta, depth, is_root):
        val = float('-inf')
        action_indexes = state.get_legal_moves()
        v_values = [0 for _ in action_indexes]
        for i, a in enumerate(action_indexes):
            c = state.copy()
            c.advance(a)
            val = max(
                val,
                alpha_beta_search(c, alpha, beta, depth - 1, predictor, original_turn, timer, False)
            )
            v_values[i] = val
            if val >= beta:
                return val
            alpha = max(alpha, val)
        if is_root:
            return v_values, action_indexes, val
        return val

    def min_value(state, alpha, beta, depth, is_root):
        val = float('inf')
        action_indexes = state.get_legal_moves()
        v_values = [0 for _ in action_indexes]
        for i, a in enumerate(action_indexes):
            c = state.copy()
            c.advance(a)
            val = min(
                val,
                alpha_beta_search(c, alpha, beta, depth - 1, predictor, original_turn, timer, False)
            )
            v_values[i] = val
            if val <= alpha:
                return val
            beta = min(beta, val)
        if is_root:
            return v_values, action_indexes, val
        return val

    # TODO: fix     or not timer.have_time_left()
    # TODO: Make optinal fixed deapth.
    # TODO: Maybe train more?
    if state.is_game_over() or depth <= 0:
        return zero_sum_2v2_evaluation(
            state=state,
            original_turn=original_turn,
            predictor=predictor
        )

    if state.turn == original_turn:
        return max_value(state, alpha, beta, depth, is_root)
    else:
        return min_value(state, alpha, beta, depth, is_root)


class AlphaBeta(BaseExpert):
    """ This implementation is limited to Zero-sum,
        two-player deterministic markov games """

    def __init__(self):
        self.timer = Timer()

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):

        timer = Timer()
        timer.start_search_timer(search_time=search_time)

        depth = 1
        v_values_last, action_indexes_last, val_last = [], [], float('-inf')
        v_values, action_indexes, val = [], [], float('-inf')
        while timer.have_time_left():
            v_values, action_indexes, val = v_values_last, action_indexes_last, val_last
            v_values_last, action_indexes_last, val_last = alpha_beta_search(
                state=state,
                alpha=float('-inf'),
                beta=float('inf'),
                depth=depth,
                predictor=predictor,
                original_turn=state.turn,
                timer=timer,
                is_root=True
            )
            depth += 1

        return v_values, action_indexes, val

