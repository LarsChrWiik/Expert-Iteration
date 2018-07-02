
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Support.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation


def minimax(state, depth, predictor, original_turn, is_root, timer=None):

    def max_value(state, depth):
        val = float('-inf')
        action_indexes = state.get_legal_moves()
        v_values = [0 for _ in action_indexes]
        for i, a in enumerate(action_indexes):
            c = state.copy()
            c.advance(a)
            val = max(
                val,
                minimax(c, depth - 1, predictor, original_turn, False, timer)
            )
            v_values[i] = val
        if is_root:
            return v_values, action_indexes, val
        return val

    def min_value(state, depth):
        val = float('inf')
        action_indexes = state.get_legal_moves()
        v_values = [0 for _ in action_indexes]
        for i, a in enumerate(action_indexes):
            c = state.copy()
            c.advance(a)
            val = min(
                val,
                minimax(c, depth - 1, predictor, original_turn, False, timer)
            )
            v_values[i] = val
        if is_root:
            return v_values, action_indexes, val
        return val

    if state.is_game_over() or depth <= 0 or \
            (not is_root and timer is not None and not timer.have_time_left()):
        """ The root will never enter this if-statement.
            This assumes that the root is never a state that is game over. """
        return zero_sum_2v2_evaluation(state, original_turn, predictor)

    if state.turn == original_turn:
        return max_value(state, depth)
    else:
        return min_value(state, depth)


class Minimax(BaseExpert):
    """ This implementation is designed fpr Zero-sum,
        two-player deterministic markov games """

    def __init__(self, fixed_depth=None):
        self.fixed_depth = fixed_depth

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):

        if self.fixed_depth is not None:
            # Fixed depth Minimax-search.
            v_values, action_indexes, val = minimax(
                state=state,
                depth=self.fixed_depth,
                predictor=predictor,
                original_turn=state.turn,
                is_root=True
            )
            return v_values, action_indexes, val
        else:
            # Iterative deepening Minimax-search.
            timer = Timer()
            timer.start_search_timer(search_time=search_time)
            v_values, action_indexes, val = None, None, None
            depth = 1
            while True:

                v_values_new, action_indexes_new, val_new = minimax(
                    state=state,
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
