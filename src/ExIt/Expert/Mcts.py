
from ExIt.Expert.BaseExpert import BaseExpert
from ExIt.Apprentice import BaseApprentice
from Games.GameLogic import BaseGame
from Misc.TrainingTimer import TrainingTimer
from ExIt.Evaluator import zero_sum_2v2_evaluation
from ExIt.ActionPolicy import explore_action, exploit_action, p_proportional, argmax
from math import sqrt
from random import shuffle


class Mcts(BaseExpert):

    def __init__(self, c):
        # Exploration parameter in UCB.
        self.c = c

    def search(self, state: BaseGame, predictor: BaseApprentice, search_time, use_off_policy):
        # Expected Q values from state s.       Q[s]   or   Q[s][a]
        Q = {}
        # Number of times state s was visited.  N[s]
        N = {}
        # Predicted P values from state s.      P[s]   or   P[s][a]
        P = {}
        # Predicted v value of state s.         V[s]
        V = {}

        original_turn = state.turn

        def mcts_search(state):

            fv = state.get_feature_vector()
            legal_moves = state.get_legal_moves()
            s = tuple(fv)

            # When unexplored child - predict and store info from this state.
            if s not in P:
                P[s] = predictor.pred_pi(X=fv)
                V[s] = zero_sum_2v2_evaluation(state, original_turn, predictor)
                N[s] = [0 for _ in range(state.num_actions)]
                Q[s] = [0 for _ in range(state.num_actions)]
                return V[s]

            # Return v value if state is game over.
            if state.is_game_over():
                return V[s]

            # Find action that maximizes Upper Confidence Bound (UCB).
            u_max = -float("inf")
            a_best = -1
            a_shuffled = list(enumerate(legal_moves))
            shuffle(a_shuffled)
            for i, a in a_shuffled:
                if N[s][a] == 0:
                    # Choose this action if it has not been tried.
                    a_best = a
                    break
                else:
                    u = Q[s][a] + self.c * P[s][a] * sqrt(sum(N[s])) / (1 + N[s][a])
                if u > u_max:
                    u_max = u
                    a_best = a
            # Action that maximizes UCB.
            a = a_best

            # Recursive call to find the v value to backpropagate.
            next_state = state.copy()
            next_state.advance(a)
            v = mcts_search(next_state)

            # Backpropagation step - update Q and N.
            Q[s][a] = (N[s][a] * Q[s][a] + v) / (N[s][a] + 1)
            N[s][a] += 1

            return v

        """ ***** SEARCH CODE ***** """

        timer = TrainingTimer(search_time)
        timer.start_new_lap()
        while timer.has_time_left() or len(N) <= 1:
            mcts_search(state)

        # Get V values and action indexes of legal moves.
        lm = state.get_legal_moves()
        s = tuple(state.get_feature_vector())
        ni = [n for i, n in enumerate(N[s]) if i in lm]
        vi = [q for i, q in enumerate(Q[s]) if i in lm]

        if use_off_policy:
            # Off-policy is the proportional of the N values.
            return explore_action(ni, lm), lm[argmax(ni)], None
        else:
            # On-policy is the action that leads to the best Q value.
            return exploit_action(vi, lm), lm[argmax(vi)], None
