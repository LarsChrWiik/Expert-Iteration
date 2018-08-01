
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Misc.TrainingTimer import TrainingTimer
from ExIt.Evaluator import zero_sum_2v2_evaluation
from ExIt.Policy import explore_proportional, explore_proportional_with_guidance, exploit_action, e_greedy, vi_proportional


class Minimax(BaseExpert):
    """ Minimax and Alpha Beta implementation.
        Set use_alpha_beta=True to use Alpha-Beta search instead of Minimax search.

        This implementation is designed for Zero-sum,
        two-player deterministic markov games """

    def __init__(self, fixed_depth=None, use_alpha_beta=False, switch=False):
        super().__init__()
        self.fixed_depth = fixed_depth
        self.alpha = float('-inf')
        self.beta = float('inf')
        self.stop_search_contradiction = True
        self.use_alpha_beta = use_alpha_beta
        self.switch = switch
        extra_name = ""
        if fixed_depth is not None:
            extra_name = "_Depth-" + str(fixed_depth)
        if switch:
            self.__name__ = "AB-Minimax" + extra_name
        else:
            if use_alpha_beta:
                self.__name__ = "AB" + extra_name
            else:
                self.__name__ = "Minimax" + extra_name

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time, always_exploit):
        # Try with switching between Minimax and AB.
        if self.switch:
            self.use_alpha_beta = not self.use_alpha_beta

        # Predicted v value of state s.         V[s]
        V = {}

        timer = None
        original_turn = state.turn

        def alpha_beta_search(state, alpha, beta, depth, is_root=False, initial_search=False):
            """ If alpha and beta is None, then this is a Minimax search """

            if not initial_search and timer is not None and not timer.has_time_left():
                raise Exception("Stop Search")

            def max_value(state, alpha, beta, depth):
                v = float('-inf')
                lm = state.get_legal_moves()
                vi = [0 for _ in lm]
                for i, a in enumerate(lm):
                    c = state.copy()
                    c.advance(a)
                    v = max(
                        v,
                        alpha_beta_search(c, alpha, beta, depth - 1, initial_search=initial_search)
                    )
                    vi[i] = v
                    if self.use_alpha_beta:
                        if v >= beta:
                            return v
                        alpha = max(alpha, v)
                if is_root:
                    return vi, v
                return v

            def min_value(state, alpha, beta, depth):
                v = float('inf')
                lm = state.get_legal_moves()
                vi = [0 for _ in lm]
                for i, a in enumerate(lm):
                    c = state.copy()
                    c.advance(a)
                    v = min(
                        v,
                        alpha_beta_search(c, alpha, beta, depth - 1, initial_search=initial_search)
                    )
                    vi[i] = v
                    if self.use_alpha_beta:
                        if v <= alpha:
                            return v
                        beta = min(beta, v)
                if is_root:
                    return vi, v
                return v

            if depth == 0 and not state.is_game_over():
                """ Disapproval of the contradiction.
                    Which means that a greater depth is needed. """
                self.stop_search_contradiction = False

            if state.is_game_over() or depth <= 0 or \
                    (not is_root and timer is not None and not timer.has_time_left()):
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

        vi, v = None, None
        if self.fixed_depth is not None:
            # Fixed depth.
            vi, v = alpha_beta_search(
                state=state,
                alpha=self.alpha,
                beta=self.beta,
                depth=self.fixed_depth,
                is_root=True
            )
        else:
            # Iterative deepening.
            timer = TrainingTimer(search_time)
            timer.start_new_lap()
            depth = 1
            initial_search = True
            while True:
                self.stop_search_contradiction = True
                try:
                    vi_new, v_new = alpha_beta_search(
                        state=state,
                        alpha=self.alpha,
                        beta=self.beta,
                        depth=depth,
                        is_root=True,
                        initial_search=initial_search
                    )
                except:
                    break
                initial_search = False
                if vi is None:
                    # This ensures that at least one iteration is stored.
                    vi, v = vi_new, v_new

                if not timer.has_time_left():
                    """ This prevents unfinished evaluation updates, which
                        ensures that the final evaluations are calculated using full AB search. """
                    break

                # Update evaluations.
                vi, v = vi_new, v_new

                if self.stop_search_contradiction:
                    # Early stopping - no greater depth is needed.
                    break

                depth += 1

        lm = state.get_legal_moves()
        best_a = exploit_action(vi, lm)
        if always_exploit:
            # Exploration-policy: Exploit - the action that leads to the highest v value.
            return best_a, best_a, v
        else:
            """ 
            ***** Potential other Exploration-policies: *****
                (proportional of VI)
                return vi_proportional(vi, lm), best_a, v

                (E-greedy)
                return e_greedy(vi, lm, e=0.1), best_a, v
            """
            # Exploration-policy: proportional of the PI with guidance of VI.
            pi = predictor.pred_pi(state.get_feature_vector())
            pi = [p for i, p in enumerate(pi) if i in lm]
            return explore_proportional_with_guidance(pi, vi, lm), best_a, v
