
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Misc.Timer import Timer
from ExIt.Evaluator import zero_sum_2v2_evaluation


class Minimax(BaseExpert):
    """ Minimax and Alpha Beta implementation.
        Set use_alpha_beta=True to use Alpha-Beta search instead of Minimax search.

        This implementation is designed for Zero-sum,
        two-player deterministic markov games """

    def __init__(self, fixed_depth=None, use_alpha_beta=False):
        self.fixed_depth = fixed_depth
        self.alpha = None if not use_alpha_beta else float('-inf')
        self.beta = None if not use_alpha_beta else float('inf')
        self.__name__ = "Minimax" if (self.alpha, self.beta) == (None, None) else "AlphaBeta"

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time: float):
        # Predicted v value of state s.         V[s]
        V = {}

        timer = None
        original_turn = state.turn

        def alpha_beta_search(state, alpha, beta, depth, is_root=False):
            """ If alpha and beta is None, then this is a Minimax search """

            def max_value(state, alpha, beta, depth):
                val = float('-inf')
                legal_moves = state.get_legal_moves()
                v_values = [0 for _ in legal_moves]
                for i, a in enumerate(legal_moves):
                    c = state.copy()
                    c.advance(a)
                    val = max(
                        val,
                        alpha_beta_search(c, alpha, beta, depth - 1)
                    )
                    v_values[i] = val
                    if (alpha, beta) != (None, None):
                        if val >= beta:
                            return val
                        alpha = max(alpha, val)
                if is_root:
                    return v_values, legal_moves, val
                return val

            def min_value(state, alpha, beta, depth):
                val = float('inf')
                legal_moves = state.get_legal_moves()
                v_values = [0 for _ in legal_moves]
                for i, a in enumerate(legal_moves):
                    c = state.copy()
                    c.advance(a)
                    val = min(
                        val,
                        alpha_beta_search(c, alpha, beta, depth - 1)
                    )
                    v_values[i] = val
                    if (alpha, beta) != (None, None):
                        if val <= alpha:
                            return val
                        beta = min(beta, val)
                if is_root:
                    return v_values, legal_moves, val
                return val

            if state.is_game_over() or depth <= 0 or \
                    (not is_root and timer is not None and not timer.have_time_left()):
                """ The root will never enter this if-statement.
                    This assumes that the root is never a state that is game over. """
                s = tuple(state.get_feature_vector())
                if s in V:
                    return V[s]
                else:
                    v = zero_sum_2v2_evaluation(state, original_turn, predictor)
                    V[s] = v
                    return v

            # The root node will always call max_value.
            if state.turn == original_turn:
                return max_value(state, alpha, beta, depth)
            else:
                return min_value(state, alpha, beta, depth)

        """ ***** SEARCH CODE ***** """

        if self.fixed_depth is not None:
            # Fixed depth.
            v_values, legal_moves, val = alpha_beta_search(
                state=state,
                alpha=self.alpha,
                beta=self.beta,
                depth=self.fixed_depth,
                is_root=True
            )
            return v_values, legal_moves, val
        else:
            # Iterative deepening.
            timer = Timer()
            timer.start_search_timer(search_time=search_time)
            v_values, legal_moves, val = None, None, None
            depth = 1
            while True:
                v_values_new, legal_moves_new, val_new = alpha_beta_search(
                    state=state,
                    alpha=self.alpha,
                    beta=self.beta,
                    depth=depth,
                    is_root=True
                )

                if v_values is None:
                    # This ensures that at least one iteration is stored.
                    v_values, legal_moves, val = v_values_new, legal_moves_new, val_new

                if not timer.have_time_left():
                    # This prevents unfinished evaluation updates, which
                    # ensures that the final evaluations are calculated using full AB search.
                    break

                # Update evaluations.
                v_values, legal_moves, val = v_values_new, legal_moves_new, val_new

                # Allow early search stop if it is not needed.
                if depth >= state.max_game_depth:
                    break

                depth += 1

            return v_values, legal_moves, val
