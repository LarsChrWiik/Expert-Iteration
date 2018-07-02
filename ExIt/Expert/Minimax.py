
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Support.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation


def alpha_beta_search(state, alpha, beta, depth, predictor, original_turn, is_root, timer=None):
    """ If alpha and beta is None, then this is a Minimax search """

    def max_value(state, alpha, beta, depth):
        val = float('-inf')
        action_indexes = state.get_legal_moves()
        v_values = [0 for _ in action_indexes]
        for i, a in enumerate(action_indexes):
            c = state.copy()
            c.advance(a)
            val = max(
                val,
                alpha_beta_search(c, alpha, beta, depth - 1, predictor, original_turn, False, timer)
            )
            v_values[i] = val
            if (alpha, beta) != (None, None):
                if val >= beta:
                    return val
                alpha = max(alpha, val)
        if is_root:
            return v_values, action_indexes, val
        return val

    def min_value(state, alpha, beta, depth):
        val = float('inf')
        action_indexes = state.get_legal_moves()
        v_values = [0 for _ in action_indexes]
        for i, a in enumerate(action_indexes):
            c = state.copy()
            c.advance(a)
            val = min(
                val,
                alpha_beta_search(c, alpha, beta, depth - 1, predictor, original_turn, False, timer)
            )
            v_values[i] = val
            if (alpha, beta) != (None, None):
                if val <= alpha:
                    return val
                beta = min(beta, val)
        if is_root:
            return v_values, action_indexes, val
        return val

    if state.is_game_over() or depth <= 0 or \
            (not is_root and timer is not None and not timer.have_time_left()):
        """ The root will never enter this if-statement.
            This assumes that the root is never a state that is game over. """
        return zero_sum_2v2_evaluation(state, original_turn, predictor)

    if state.turn == original_turn:
        return max_value(state, alpha, beta, depth)
    else:
        return min_value(state, alpha, beta, depth)


class Minimax(BaseExpert):
    """ This implementation is designed for Zero-sum,
        two-player deterministic markov games """

    def __init__(self, fixed_depth=None, use_alpha_beta=False):
        self.fixed_depth = fixed_depth
        self.alpha = None if not use_alpha_beta else float('-inf')
        self.beta = None if not use_alpha_beta else float('inf')

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):

        if self.fixed_depth is not None:
            # Fixed depth.
            v_values, action_indexes, val = alpha_beta_search(
                state=state,
                alpha=self.alpha,
                beta=self.beta,
                depth=self.fixed_depth,
                predictor=predictor,
                original_turn=state.turn,
                is_root=True
            )
            return v_values, action_indexes, val
        else:
            # Iterative deepening.
            timer = Timer()
            timer.start_search_timer(search_time=search_time)
            v_values, action_indexes, val = None, None, None
            depth = 1
            while True:
                v_values_new, action_indexes_new, val_new = alpha_beta_search(
                    state=state,
                    alpha=self.alpha,
                    beta=self.beta,
                    depth=depth,
                    predictor=predictor,
                    original_turn=state.turn,
                    timer=timer,
                    is_root=True
                )

                if v_values is None:
                    # This ensures that at least one iteration is stored.
                    v_values, action_indexes, val = v_values_new, action_indexes_new, val_new

                if not timer.have_time_left():
                    # This prevents unfinished evaluation updates, which
                    # ensures that the final evaluations are calculated using full AB search.
                    break

                # Update evaluations.
                v_values, action_indexes, val = v_values_new, action_indexes_new, val_new

                # Allow early search stop if it is not needed.
                if depth >= state.max_game_depth:
                    break

                depth += 1

            return v_values, action_indexes, val
